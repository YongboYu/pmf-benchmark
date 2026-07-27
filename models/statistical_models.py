import logging
import time
from pathlib import Path
from typing import Any

import wandb
import yaml
from darts.models import (
    ARIMA,
    TBATS,
    AutoARIMA,
    ExponentialSmoothing,
    FourTheta,
    Prophet,
    Theta,
)
from darts.utils.utils import ModelMode, SeasonalityMode
from utils.data_loader import DataLoader
from utils.evaluation import Evaluator
from utils.wandb_logger import WandbLogger

# Initialize logger
logger = logging.getLogger(__name__)


class StatisticalModels:
    def __init__(self, config: dict[str, Any] | str, model_name: str | None = None):
        """Initialize statistical models.

        Args:
            config: Either config dictionary or path to base config file
            model_name: Optional name of a specific model to initialize
        """
        # Load configurations
        if isinstance(config, dict):
            self.base_config = config
            self.model_config = config["model_configs"]["statistical_models"]
        else:
            with open(config) as f:
                self.base_config = yaml.safe_load(f)
            with open(
                Path(config).parent / "model_configs" / "statistical_models.yaml"
            ) as f:
                self.model_config = yaml.safe_load(f)

        # Initialize components
        self.data_loader = DataLoader(self.base_config)
        self.evaluator = Evaluator(self.base_config)

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

    def _initialize_model(self, model_name: str, model_info: dict[str, Any]):
        """Initialize statistical model with parameters"""
        params = model_info.get("params", {})

        try:
            if model_name == "ar2" or model_name == "arima":
                return ARIMA(**params)
            elif model_name == "auto_arima":
                return AutoARIMA(**params)
            elif model_name == "exp_smoothing":
                # Convert string parameters to ModelMode enums
                if "trend" in params:
                    if params["trend"] == "additive":
                        params["trend"] = ModelMode.ADDITIVE
                    elif params["trend"] == "multiplicative":
                        params["trend"] = ModelMode.MULTIPLICATIVE
                    else:
                        params["trend"] = ModelMode.NONE
                if "seasonal" in params:
                    if params["seasonal"] == "additive":
                        params["seasonal"] = SeasonalityMode.ADDITIVE
                    elif params["seasonal"] == "multiplicative":
                        params["seasonal"] = SeasonalityMode.MULTIPLICATIVE
                    else:
                        params["seasonal"] = SeasonalityMode.NONE
                return ExponentialSmoothing(**params)
            elif model_name == "tbats":
                return TBATS(**params)
            elif model_name == "theta":
                if "season_mode" in params:
                    params["season_mode"] = (
                        SeasonalityMode.ADDITIVE
                        if params["season_mode"] == "additive"
                        else SeasonalityMode.MULTIPLICATIVE
                    )
                return Theta(**params)
            elif model_name == "four_theta":
                if "season_mode" in params:
                    params["season_mode"] = (
                        SeasonalityMode.ADDITIVE
                        if params["season_mode"] == "additive"
                        else SeasonalityMode.MULTIPLICATIVE
                    )
                if "model_mode" in params:
                    params["model_mode"] = (
                        ModelMode.ADDITIVE
                        if params["model_mode"] == "additive"
                        else ModelMode.MULTIPLICATIVE
                    )
                return FourTheta(**params)
            elif model_name == "prophet":
                return Prophet(**params)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing {model_name}: {e!s}")
            raise

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
        """Train model and generate predictions using expanding window approach"""
        logger.info(f"Training {model_name} model...")
        self.wandb_logger = wandb_logger

        if not self.models[model_name]:
            logger.info(f"Model {model_name} not found")
            return {}

        model_info = self.model_config["models"].get(model_name, {})
        model_params = model_info.get("params", {})

        if self.wandb_logger is not None:
            wandb.config.update({"model_params": {model_name: model_params}})

        try:
            # Create expanding window test dataset
            test_input_seq, test_output_seq = self.data_loader.create_expanding_io_data(
                train=train, val=val, test=test, horizon=horizon
            )

            start_time = time.time()

            # Train separate model for each component
            all_component_predictions = []
            trained_models = []

            for component in train.components:
                component_predictions = []

                # Create new model instance for each component
                component_model = self._initialize_model(
                    model_name, self.model_config["models"][model_name]
                )

                # For each input sequence in the expanding window
                for input_seq in test_input_seq:
                    # Extract component data and fit model
                    input_seq_component = input_seq[component]
                    component_model.fit(input_seq_component)

                    # Generate predictions
                    pred = component_model.predict(n=horizon)
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

            # Calculate training time
            training_time = time.time() - start_time
            wandb.log({"training_time": training_time})

            # # Log metrics if wandb_logger is available
            # if wandb_logger:
            #     wandb_logger.log_metrics({
            #         'training_time': training_time,
            #         'n_components': len(train.components),
            #         'model_type': model_name,
            #         'horizon': horizon,
            #         'dataset': dataset
            #     }, prefix=model_name)
            #
            #     # Log model artifacts
            #     wandb_logger.log_model_artifacts(model_name, {
            #         'model_type': model_name,
            #         'n_components': len(train.components),
            #         'training_time': training_time,
            #         'dataset': dataset,
            #         'horizon': horizon
            #     })

            return {
                "predictions": all_predictions,
                "actuals": test_output_seq,
                "model": trained_models,
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
        """Get list of enabled model names"""
        return list(self.models.keys())
