import logging
import time
from typing import Any

import numpy as np
import optuna
import wandb
import yaml
from darts.metrics import mae
from darts.models import RandomForest, XGBModel
from darts.timeseries import TimeSeries
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import (
    plot_optimization_history,
)
from utils.data_loader import DataLoader
from utils.evaluation import Evaluator
from utils.optuna_manager import OptunaManager
from utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


class CovariateRegressionModels:
    def __init__(self, config: dict[str, Any] | str):
        """Initialize covariate regression models with configuration

        Args:
            config: Either config dictionary or path to config file
        """
        # Load configurations
        if isinstance(config, dict):
            self.config = config["model_configs"]["covariate_regression_models"]
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
            "cov_random_forest": {
                "model": RandomForest,
                "enabled": self.config["models"]["random_forest"]["enabled"],
                "params": self.config["models"]["random_forest"][
                    "hyperparameter_ranges"
                ],
            },
            "cov_xgboost": {
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

        # Always include lags for all models (both target and covariates)
        params["lags"] = trial.suggest_categorical("lags", possible_lags)
        params["lags_past_covariates"] = params["lags"]  # Use same lags for covariates

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

    def _prepare_covariates(
        self, series: TimeSeries, target_component: str
    ) -> TimeSeries:
        """Prepare covariates by stacking all other components except the target"""
        covariate_components = [
            comp for comp in series.components if comp != target_component
        ]
        if not covariate_components:
            return None

        covariates = series[covariate_components[0]]
        for comp in covariate_components[1:]:
            covariates = covariates.stack(series[comp])

        return covariates

    def objective(
        self,
        trial: optuna.Trial,
        model_name: str,
        model_class,
        train_component: TimeSeries,
        train_covariates: TimeSeries,
        val_component: TimeSeries | None,
        val_covariates: TimeSeries | None,
        horizon: int,
    ) -> float:
        """Optuna objective function for hyperparameter optimization"""
        try:
            params = self._get_model_params(trial, model_name, horizon)
            model = model_class(**params)
            model.fit(train_component, past_covariates=train_covariates)

            if val_component is None:
                # If no validation set, use a portion of training data
                train_size = int(len(train_component) * 0.8)
                val_component = train_component[train_size:]
                val_covariates = train_covariates[train_size:]
                train_component_subset = train_component[:train_size]
                train_covariates_subset = train_covariates[:train_size]
                model.fit(
                    train_component_subset, past_covariates=train_covariates_subset
                )

            # Create seq2seq validation dataset
            val_input_seq, val_output_seq = self.data_loader.create_seq2seq_io_data(
                train=train_component,
                val=val_component,
                test=None,
                input_length=params["lags"],
                output_length=horizon,
                dataset_type="val",
            )

            # Create corresponding covariate sequences
            val_cov_seq, _ = self.data_loader.create_seq2seq_io_data(
                train=train_covariates,
                val=val_covariates,
                test=None,
                input_length=params["lags"],
                output_length=horizon,
                dataset_type="val",
            )

            # Generate predictions
            val_predictions = []
            for input_seq, cov_seq in zip(val_input_seq, val_cov_seq, strict=False):
                pred = model.predict(
                    n=horizon, series=input_seq, past_covariates=cov_seq
                )
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
        """Train separate models for each component using other components as covariates"""
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

            for target_component in train.components:
                logger.info(
                    f"Training covariate {model_name} for target: {target_component}"
                )

                # Prepare covariates for each dataset split
                train_covariates = self._prepare_covariates(train, target_component)
                val_covariates = (
                    self._prepare_covariates(val, target_component)
                    if val is not None
                    else None
                )
                test_covariates = self._prepare_covariates(test, target_component)

                # Create study for this component
                component_study = self.optuna_manager.create_study(
                    dataset=f"{dataset}_{target_component}",
                    horizon=horizon,
                    model_group="covariate_regression",
                    model_name=f"cov_{model_name}",
                )

                wandb_callback = WeightsAndBiasesCallback(
                    metric_name=f"val_mae_{target_component}", wandb_kwargs=wandb_kwargs
                )

                # Optimize for this component
                component_study.optimize(
                    func=lambda trial, target_component=target_component, train_covariates=train_covariates, val_covariates=val_covariates: (
                        self.objective(
                            trial,
                            model_name,
                            self.models[model_name]["model"],
                            train[target_component],
                            train_covariates,
                            val[target_component] if val is not None else None,
                            val_covariates,
                            horizon,
                        )
                    ),
                    n_trials=self.config["common"]["n_trials"],
                    callbacks=[wandb_callback],
                )

                # Log optimization results
                try:
                    wandb.log(
                        {
                            f"cov_{model_name}/{target_component}/optimization_history": plot_optimization_history(
                                component_study
                            )
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not plot optimization history for {target_component}: {e!s}"
                    )

                # Train final model with best parameters
                best_params = component_study.best_params
                best_params["output_chunk_length"] = horizon
                best_params["lags_past_covariates"] = best_params["lags"]

                # Train on combined data
                if val is not None:
                    combined_train_component = train[target_component].concatenate(
                        val[target_component]
                    )
                    combined_train_covariates = train_covariates.concatenate(
                        val_covariates
                    )
                else:
                    combined_train_component = train[target_component]
                    combined_train_covariates = train_covariates

                component_model = self.models[model_name]["model"](**best_params)
                component_model.fit(
                    combined_train_component, past_covariates=combined_train_covariates
                )

                # Generate predictions
                test_input_seq, test_output_seq = (
                    self.data_loader.create_seq2seq_io_data(
                        train=train[target_component],
                        val=val[target_component] if val is not None else None,
                        test=test[target_component],
                        input_length=best_params["lags"],
                        output_length=horizon,
                        dataset_type="test",
                    )
                )

                test_cov_seq, _ = self.data_loader.create_seq2seq_io_data(
                    train=train_covariates,
                    val=val_covariates if val is not None else None,
                    test=test_covariates,
                    input_length=best_params["lags"],
                    output_length=horizon,
                    dataset_type="test",
                )

                component_predictions = []
                for input_seq, cov_seq in zip(
                    test_input_seq, test_cov_seq, strict=False
                ):
                    pred = component_model.predict(
                        n=horizon, series=input_seq, past_covariates=cov_seq
                    )
                    component_predictions.append(pred)

                all_component_predictions.append(component_predictions)
                trained_models.append(component_model)

            # Combine predictions from all components
            all_predictions = []
            for seq_idx in range(len(test_input_seq)):
                seq_predictions = [
                    comp_preds[seq_idx] for comp_preds in all_component_predictions
                ]
                combined_pred = seq_predictions[0]
                for pred in seq_predictions[1:]:
                    combined_pred = combined_pred.stack(pred)
                all_predictions.append(combined_pred)

            total_training_time = time.time() - training_start_time
            wandb.log({f"cov_{model_name}_training_time": total_training_time})

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
                "model_name": f"cov_{model_name}",
            }

        except Exception as e:
            error_msg = f"Error training covariate {model_name}: {e!s}"
            logger.error(error_msg)
            raise

    def get_model_names(self) -> list:
        """Return list of enabled model names"""
        return [name for name, info in self.models.items() if info["enabled"]]

    def set_covariates(
        self,
        train_covariates: TimeSeries,
        val_covariates: TimeSeries | None,
        test_covariates: TimeSeries,
    ):
        """Set covariate data for the model"""
        self.train_covariates = train_covariates
        self.val_covariates = val_covariates
        self.test_covariates = test_covariates
