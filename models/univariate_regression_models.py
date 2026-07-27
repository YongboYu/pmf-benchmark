import logging
import time
from typing import Any

import numpy as np
import optuna
import wandb
import yaml
from darts.metrics import mae
from darts.models import RandomForest, XGBModel
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import (
    plot_optimization_history,
)
from utils.data_loader import DataLoader
from utils.evaluation import Evaluator
from utils.optuna_manager import OptunaManager
from utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


class UnivariateRegressionModels:
    def __init__(self, config: dict[str, Any] | str):
        """Initialize univariate regression models with configuration

        Args:
            config: Either config dictionary or path to config file
        """
        # Load configurations
        if isinstance(config, dict):
            self.config = config["model_configs"]["univariate_regression_models"]
            self.full_config = config
        else:
            with open(config) as f:
                self.config = yaml.safe_load(f)
                self.full_config = self.config

        # Initialize utilities
        self.optuna_manager = OptunaManager(self.full_config)
        self.data_loader = DataLoader(self.full_config)
        self.evaluator = Evaluator(self.full_config)

        # Map model names to their Darts implementations
        self.models = {
            "uni_random_forest": {
                "model": RandomForest,
                "enabled": self.config["models"]["random_forest"]["enabled"],
                "params": self.config["models"]["random_forest"][
                    "hyperparameter_ranges"
                ],
            },
            "uni_xgboost": {
                "model": XGBModel,
                "enabled": self.config["models"]["xgboost"]["enabled"],
                "params": self.config["models"]["xgboost"]["hyperparameter_ranges"],
            },
        }

        self.wandb_logger = None

    def _get_model_params(
        self, trial: optuna.Trial, model_name: str, horizon: int
    ) -> dict[str, Any]:
        """Get hyperparameters for a specific model from Optuna trial"""
        params = {}
        param_ranges = self.models[model_name]["params"]

        # Get horizon-specific lags configuration
        horizon_key = f"horizon_{horizon}"
        lags_config = self.config["common"]["lags"][horizon_key]

        # Get lags as a hyperparameter to optimize
        min_lag = lags_config["min_lag"]
        max_lag = lags_config["max_lag"]
        step_size = lags_config["step_size"]

        # Create list of possible lag values
        possible_lags = list(range(min_lag, max_lag + 1, step_size))

        # Always include lags for all models
        params["lags"] = trial.suggest_categorical("lags", possible_lags)

        # Always add output_chunk_length parameter (not optimized)
        params["output_chunk_length"] = horizon

        # Get model-specific hyperparameters
        if param_ranges:
            for param_name, param_values in param_ranges.items():
                if isinstance(param_values, list):
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_values
                    )
                elif isinstance(param_values, tuple):
                    if any(isinstance(v, float) for v in param_values):
                        params[param_name] = trial.suggest_float(
                            param_name, *param_values
                        )
                    else:
                        params[param_name] = trial.suggest_int(
                            param_name, *param_values
                        )

        return params

    def objective(
        self,
        trial: optuna.Trial,
        model_name: str,
        model_class,
        train_component,
        val_component,
        horizon: int,
    ) -> float:
        """Optuna objective function for hyperparameter optimization"""
        try:
            params = self._get_model_params(trial, model_name, horizon)
            model = model_class(**params)
            model.fit(train_component)

            # Create seq2seq validation dataset
            val_input_seq, val_output_seq = self.data_loader.create_seq2seq_io_data(
                train=train_component,
                val=val_component,
                test=None,
                input_length=params["lags"],
                output_length=horizon,
                dataset_type="val",
            )

            # Generate predictions
            val_predictions = []
            for input_seq in val_input_seq:
                pred = model.predict(n=horizon, series=input_seq)
                val_predictions.append(pred)

            # Calculate average MAE
            val_scores = []
            for pred, true in zip(val_predictions, val_output_seq, strict=False):
                val_scores.append(mae(true, pred))
            val_score = np.mean(val_scores)

            wandb.log({"val_mae": val_score})

            return val_score

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e!s}")
            raise optuna.exceptions.TrialPruned() from e

    def train_and_predict(
        self,
        model_name: str,
        train,
        val,
        test,
        transformer,
        horizon: int,
        dataset: str,
        study: optuna.Study | None = None,
        wandb_logger: WandbLogger = None,
    ) -> dict[str, Any]:
        """Train separate models for each component and generate predictions"""
        try:
            self.wandb_logger = wandb_logger

            if not self.models[model_name]["enabled"]:
                logger.info(f"Model {model_name} is disabled in config")
                return {}

            training_start_time = time.time()
            wandb_kwargs = self.config.get("wandb", {}).copy()

            # Train separate model for each component
            all_component_predictions = []
            trained_models = []

            for component in train.components:
                logger.info(
                    f"Training univariate {model_name} for component: {component}"
                )

                # Create study for this component
                component_study = self.optuna_manager.create_study(
                    dataset=f"{dataset}_{component}",
                    horizon=horizon,
                    model_group="univariate_regression",
                    model_name=f"uni_{model_name}",
                )

                wandb_callback = WeightsAndBiasesCallback(
                    metric_name=f"val_mae_{component}", wandb_kwargs=wandb_kwargs
                )

                # Optimize for this component
                component_study.optimize(
                    func=lambda trial, component=component: self.objective(
                        trial,
                        model_name,
                        self.models[model_name]["model"],
                        train[component],
                        val[component],
                        horizon,
                    ),
                    n_trials=self.config["common"]["n_trials"],
                    callbacks=[wandb_callback],
                )

                # Log optimization results
                try:
                    wandb.log(
                        {
                            f"uni_{model_name}/{component}/optimization_history": plot_optimization_history(
                                component_study
                            )
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not plot optimization history for {component}: {e!s}"
                    )

                # Train final model with best parameters
                best_params = component_study.best_params
                best_params["output_chunk_length"] = horizon

                # Train on combined data
                combined_train_component = (
                    train[component].concatenate(val[component])
                    if val is not None
                    else train[component]
                )
                component_model = self.models[model_name]["model"](**best_params)
                component_model.fit(combined_train_component)

                component_predictions = []
                test_input_seq, test_output_seq = (
                    self.data_loader.create_seq2seq_io_data(
                        train=train[component],
                        val=val[component] if val is not None else None,
                        test=test[component],
                        input_length=best_params["lags"],
                        output_length=horizon,
                        dataset_type="test",
                    )
                )

                for input_seq in test_input_seq:
                    pred = component_model.predict(n=horizon, series=input_seq)
                    component_predictions.append(pred)

                all_component_predictions.append(component_predictions)
                trained_models.append(component_model)

            # Modified prediction combination logic - following statistical_models.py approach
            all_predictions = []
            for seq_idx in range(len(test_input_seq)):
                seq_predictions = [
                    comp_preds[seq_idx] for comp_preds in all_component_predictions
                ]
                # Use stack method like in statistical_models.py
                combined_pred = seq_predictions[0]
                for pred in seq_predictions[1:]:
                    combined_pred = combined_pred.stack(pred)
                all_predictions.append(combined_pred)

            total_training_time = time.time() - training_start_time
            wandb.log({f"uni_{model_name}_training_time": total_training_time})

            test_input_seq, test_output_seq = self.data_loader.create_seq2seq_io_data(
                train=train,
                val=val,
                test=test,
                input_length=56,
                output_length=horizon,
                dataset_type="test",
            )

            return {
                "predictions": all_predictions,
                "actuals": test_output_seq,
                "model": trained_models,
                "training_time": total_training_time,
                "model_name": f"uni_{model_name}",
            }

        except Exception as e:
            error_msg = f"Error training univariate {model_name}: {e!s}"
            logger.error(error_msg)
            raise

    def get_model_names(self) -> list:
        """Return list of enabled model names"""
        return [name for name, info in self.models.items() if info["enabled"]]
