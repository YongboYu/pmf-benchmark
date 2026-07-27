import logging
import time
from typing import Any

import numpy as np
import optuna
import wandb
import yaml
from darts.metrics import mae
from darts.models import LightGBMModel, LinearRegressionModel, RandomForest, XGBModel
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from utils.data_loader import DataLoader
from utils.evaluation import Evaluator
from utils.optuna_manager import OptunaManager
from utils.wandb_logger import WandbLogger

# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from lightgbm import LGBMRegressor

logger = logging.getLogger(__name__)


class RegressionModels:
    def __init__(self, config: dict[str, Any] | str):
        """Initialize regression models with configuration

        Args:
            config: Either config dictionary or path to config file
        """
        # Load configurations
        if isinstance(config, dict):
            self.config = config["model_configs"]["regression_models"]
            self.full_config = config  # Store full config for OptunaManager
        else:
            with open(config) as f:
                self.config = yaml.safe_load(f)
                self.full_config = self.config

        # Initialize OptunaManager
        self.optuna_manager = OptunaManager(self.full_config)

        # Initialize DataLoader
        self.data_loader = DataLoader(self.full_config)

        # Initialize evaluator
        self.evaluator = Evaluator(self.full_config)

        # Map model names to their Darts implementations
        self.models = {
            "linear": {
                "model": LinearRegressionModel,
                "enabled": self.config["models"]["linear"]["enabled"],
                "params": {},
            },
            "random_forest": {
                "model": RandomForest,
                "enabled": self.config["models"]["random_forest"]["enabled"],
                "params": self.config["models"]["random_forest"][
                    "hyperparameter_ranges"
                ],
            },
            "xgboost": {
                "model": XGBModel,
                "enabled": self.config["models"]["xgboost"]["enabled"],
                "params": self.config["models"]["xgboost"]["hyperparameter_ranges"],
            },
            "lightgbm": {
                "model": LightGBMModel,
                "enabled": self.config["models"]["lightgbm"]["enabled"],
                "params": self.config["models"]["lightgbm"]["hyperparameter_ranges"],
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
        train,
        val,
        horizon: int,
    ) -> float:
        """Optuna objective function for hyperparameter optimization"""
        try:
            params = self._get_model_params(trial, model_name, horizon)
            model = model_class(**params)
            model.fit(train)

            # Create seq2seq validation dataset
            val_input_seq, val_output_seq = self.data_loader.create_seq2seq_io_data(
                train=train,
                val=val,
                test=None,
                input_length=params["lags"],
                output_length=horizon,
                dataset_type="val",
            )

            # Compute validation score using ground truth inputs
            val_predictions = []
            for input_seq in val_input_seq:
                pred = model.predict(n=horizon, series=input_seq)
                val_predictions.append(pred)

            # Calculate average MAE across all sequences
            val_scores = []
            for pred, true in zip(val_predictions, val_output_seq, strict=False):
                val_scores.append(mae(true, pred))
            val_score = np.mean(val_scores)

            wandb.log({"val_mae": val_score})
            # Log metrics and parameters
            # if self.wandb_logger is not None:
            # self.wandb_logger.log_trial_metrics(
            #     metrics={'mae': val_score},
            #     params=params,
            #     model_name=model_name
            # )

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
        """Train model and generate predictions using seq2seq approach"""
        try:
            # Set wandb_logger as instance attribute
            self.wandb_logger = wandb_logger

            if not self.models[model_name]["enabled"]:
                logger.info(f"Model {model_name} is disabled in config")
                return {}

            # Start timing the entire training process
            training_start_time = time.time()

            # Get wandb initialization settings from config
            wandb_kwargs = self.config.get("wandb", {}).copy()
            # Update run name to include model specific information
            # wandb_kwargs['name'] = f"{wandb_kwargs.get('name', '')}_{model_name}_{dataset}_h{horizon}"

            # Add wandb config logging here
            if wandb_logger is not None:
                # with wandb.init(**wandb_kwargs):
                wandb.config.update(
                    {
                        "model_params": {model_name: self.models[model_name]["params"]},
                        "common_params": {
                            "lags": self.config["common"]["lags"],
                            "n_trials": self.config["common"]["n_trials"],
                        },
                    }
                )

                # Create study and run optimization
                if study is None:
                    study = self.optuna_manager.create_study(
                        dataset=dataset,
                        horizon=horizon,
                        model_group="regression",
                        model_name=model_name,
                    )

                wandb_callback = WeightsAndBiasesCallback(
                    metric_name="val_mae", wandb_kwargs=wandb_kwargs
                )
                # # Get wandb callback if logger is provided
                # wandb_callback = wandb_logger.log_optuna_study(
                #     study,
                #     model_name,
                #     dataset,
                #     horizon,
                #     wandb_kwargs=wandb_kwargs
                # )

                # Run optimization with callback
                optimize_args = {
                    "func": lambda trial: self.objective(
                        trial,
                        model_name,
                        self.models[model_name]["model"],
                        train,
                        val,
                        horizon,
                    ),
                    "n_trials": self.config["common"]["n_trials"],
                    "catch": (Exception,),
                    "callbacks": [wandb_callback] if wandb_callback else None,
                }

                study.optimize(**optimize_args)

                # Log study results after optimization
                # if wandb_logger is not None:
                #     wandb_logger.log_study_results(study, model_name, dataset, horizon)
                wandb.log({"optimization_history": plot_optimization_history(study)})
                wandb.log({"param_importances": plot_param_importances(study)})
                wandb.log({"parallel_coordinate": plot_parallel_coordinate(study)})

                # Get the best parameters if available
                if study.best_trial is not None:
                    best_params = study.best_params
                    best_params["output_chunk_length"] = horizon

                    if "lags" not in best_params:
                        best_params["lags"] = study.best_trial.params.get("lags")

                    # Train final model on combined data
                    combined_train = (
                        train.concatenate(val) if val is not None else train
                    )
                    model = self.models[model_name]["model"](**best_params)

                    # Time the model fitting specifically
                    model.fit(combined_train)

                    # Calculate total training time (including optimization)
                    total_training_time = time.time() - training_start_time
                    wandb.log({"training_time": total_training_time})

                    # Create seq2seq test dataset for final evaluation
                    test_input_seq, test_output_seq = (
                        self.data_loader.create_seq2seq_io_data(
                            train=train,
                            val=val,
                            test=test,
                            input_length=best_params["lags"],
                            output_length=horizon,
                            dataset_type="test",
                        )
                    )

                    # Generate final predictions
                    all_predictions = []
                    for input_seq in test_input_seq:
                        pred = model.predict(n=horizon, series=input_seq)
                        all_predictions.append(pred)

                    # Log final results
                    wandb.run.summary.update(
                        {"best_params": best_params, "best_val_value": study.best_value}
                    )
                    # wandb.log({
                    #     f"regression_{model_name}_best_params": best_params,
                    #     f"regression_{model_name}_trained": True,
                    #     f"regression_{model_name}_n_trials": len(study.trials),
                    #     f"regression_{model_name}_best_value": study.best_value,
                    #     f"regression_{model_name}_fit_time": fit_time,
                    #     f"regression_{model_name}_total_training_time": total_training_time
                    # })

                    return {
                        "predictions": all_predictions,
                        "actuals": test_output_seq,
                        "model": model,
                        "best_params": best_params,
                        "study": study,
                        # 'fit_time': fit_time,
                        "training_time": total_training_time,
                    }

                # Return empty results if no successful trials
                return {
                    "predictions": None,
                    "actuals": None,
                    "ground_truth": None,
                    "model": None,
                    "best_params": {},
                    "study": study,
                }

        except Exception as e:
            error_msg = f"Error training {model_name}: {e!s}"
            logger.error(error_msg)
            # if wandb_logger is not None:
            #     wandb.log({
            #         f"regression_{model_name}_error": error_msg,
            #         f"regression_{model_name}_failed": True
            #     })
            raise

    def get_model_names(self) -> list:
        """Return list of enabled model names"""
        return [name for name, info in self.models.items() if info["enabled"]]
