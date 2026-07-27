import logging
import time
from pathlib import Path
from typing import Any

import wandb
import yaml
from darts.models import NaiveDrift, NaiveMean, NaiveMovingAverage, NaiveSeasonal
from utils.data_loader import DataLoader
from utils.wandb_logger import WandbLogger

# Initialize logger
logger = logging.getLogger(__name__)


class BaselineModels:
    def __init__(self, config: dict[str, Any] | str, model_name: str | None = None):
        """Initialize baseline models.

        Args:
            config: Either config dictionary or path to base config file
            model_name: Optional name of a specific model to initialize
        """
        # Load configurations
        if isinstance(config, dict):
            self.base_config = config
            self.model_config = config["model_configs"]["baseline_models"]
        else:
            with open(config) as f:
                self.base_config = yaml.safe_load(f)
            with open(
                Path(config).parent / "model_configs" / "baseline_models.yaml"
            ) as f:
                self.model_config = yaml.safe_load(f)

        # Initialize DataLoader
        self.data_loader = DataLoader(self.base_config)

        # Initialize models
        self.models = {}
        if model_name:
            # Initialize single model if specified
            model_info = self.model_config["models"].get(model_name)
            if not model_info or not model_info["enabled"]:
                raise ValueError(f"Model {model_name} not found or not enabled")
            self.models[model_name] = self._initialize_model(model_name, model_info)
        else:
            # Initialize all enabled models
            for model_name, model_info in self.model_config["models"].items():
                if model_info["enabled"]:
                    self.models[model_name] = self._initialize_model(
                        model_name, model_info
                    )

        self.training_time = 0

    def _initialize_model(self, model_name: str, model_info: dict[str, Any]):
        """Initialize baseline model with or without parameters"""
        params = model_info.get("params", {})

        # if self.wandb_logger is not None:
        #     wandb.config.update({
        #         "model_params": {
        #             model_name: params
        #         }
        #     })

        if model_name == "naive_mean":
            return NaiveMean()
        elif model_name == "persistence" or model_name == "naive_seasonal":
            return NaiveSeasonal(**params)
        elif model_name == "naive_drift":
            return NaiveDrift()
        elif model_name == "naive_moving_average":
            return NaiveMovingAverage(**params)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def train_and_predict(
        self,
        model_name: str,
        train,
        val,
        test,
        transformer,
        horizon: int,
        dataset: str,
        study: Any | None = None,
        wandb_logger: WandbLogger = None,
    ) -> dict[str, Any]:
        """Train models and generate predictions using expanding window approach"""
        # Set wandb_logger as instance attribute
        self.wandb_logger = wandb_logger

        if not self.models[model_name]:
            logger.info(f"Model {model_name} not found")
            return {}

        # Get model configuration parameters
        model_info = self.model_config["models"].get(model_name, {})
        model_params = model_info.get("params", {})

        # Update wandb config with just the model parameters in a single dictionary
        if self.wandb_logger is not None:
            wandb.config.update({"model_params": {model_name: model_params}})

        # Log that we're starting to train this model
        # wandb.log({f"baseline_training_model": model_name})

        try:
            model = self.models[model_name]
            # Create expanding window test dataset for final evaluation
            test_input_seq, test_output_seq = self.data_loader.create_expanding_io_data(
                train=train, val=val, test=test, horizon=horizon
            )

            start_time = time.time()

            # Generate predictions using expanding window inputs
            all_predictions = []
            for input_seq in test_input_seq:
                # Fit model on current input sequence
                model.fit(input_seq)
                # Generate prediction
                pred = model.predict(n=horizon)
                all_predictions.append(pred)

            # Calculate training time
            training_time = time.time() - start_time

            wandb.log({"training_time": training_time})

            # Log metrics if wandb_logger is available
            # if wandb_logger:
            # wandb_logger.log_metrics({
            #     'training_time': training_time,
            #     'model_type': str(type(model).__name__),
            #     'horizon': horizon,
            #     'dataset': dataset
            # }, prefix=model_name)
            #
            # # Log model artifacts
            # wandb_logger.log_model_artifacts(model_name, {
            #     'model_type': str(type(model).__name__),
            #     'training_time': training_time,
            #     'dataset': dataset,
            #     'horizon': horizon
            # })

            return {
                "predictions": all_predictions,
                "actuals": test_output_seq,
                "model": model,
                "training_time": training_time,
                "model_name": model_name,
            }

        except Exception as e:
            error_msg = f"Error training {model_name}: {e!s}"
            logger.error(error_msg)
            # if wandb_logger:
            #     wandb_logger.log_metrics({
            #         "error": str(e),
            #         "failed": True
            #     }, prefix=model_name)
            raise

    def get_model_names(self) -> list[str]:
        """Get list of enabled model names."""
        return list(self.models.keys())
