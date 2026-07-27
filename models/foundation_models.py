import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
import yaml
from darts import TimeSeries
from nixtla import NixtlaClient
from utils.data_loader import DataLoader
from utils.evaluation import Evaluator
from utils.wandb_logger import WandbLogger

# Initialize logger
logger = logging.getLogger(__name__)


class FoundationModels:
    def __init__(self, config: dict[str, Any] | str, model_name: str | None = None):
        """Initialize foundation models.

        Args:
            config: Either config dictionary or path to base config file
            model_name: Optional name of a specific model to initialize
        """
        # Load configurations
        if isinstance(config, dict):
            self.base_config = config
            self.model_config = config["model_configs"]["foundation_models"]
        else:
            with open(config) as f:
                self.base_config = yaml.safe_load(f)
            with open(
                Path(config).parent / "model_configs" / "foundation_models.yaml"
            ) as f:
                self.model_config = yaml.safe_load(f)

        # Initialize components
        self.data_loader = DataLoader(self.base_config)
        self.evaluator = Evaluator(self.base_config)

        # Initialize models
        self.models = {}
        if model_name:
            model_info = self.model_config["models"].get(model_name)
            if not model_info or not model_info["enabled"]:
                raise ValueError(f"Model {model_name} not found or not enabled")
            self.models[model_name] = self._initialize_model(model_name, model_info)
        else:
            for model_name, model_info in self.model_config["models"].items():
                if model_info["enabled"]:
                    self.models[model_name] = self._initialize_model(
                        model_name, model_info
                    )

    def _initialize_model(self, model_name: str, model_info: dict[str, Any]):
        """Initialize foundation model with parameters"""
        params = model_info.get("params", {})

        try:
            if model_name == "timegpt":
                return NixtlaClient(**params)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except Exception as e:
            logger.error(f"Error initializing {model_name}: {e!s}")
            raise

    def _timeseries_to_df(self, ts: TimeSeries) -> pd.DataFrame:
        """Convert TimeSeries object to DataFrame format required by TimeGPT"""
        df = ts.pd_dataframe()
        df.index.name = "timestamp"
        df.reset_index(inplace=True)
        return df

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

        # wandb.log({f"foundation_training_model": model_name})
        start_time = time.time()

        try:
            # Get expanding window test dataset
            test_input_seq, test_output_seq = self.data_loader.create_expanding_io_data(
                train=train, val=val, test=test, horizon=horizon
            )

            # Initialize TimeGPT model
            model = self.models[model_name]
            all_predictions = []

            # For each input sequence in expanding window
            for input_seq in test_input_seq:
                # Convert TimeSeries to DataFrame
                input_df = self._timeseries_to_df(input_seq)

                # Generate predictions for each component
                predictions = []
                for component in train.components:
                    # Make forecast using TimeGPT
                    fcst_df = model.forecast(
                        df=input_df,
                        h=horizon,
                        time_col="timestamp",
                        target_col=component,
                        freq=input_seq.freq_str,
                    )
                    predictions.append(fcst_df)

                # Combine predictions and convert back to TimeSeries
                combined_df = pd.concat(
                    [pred.set_index("timestamp") for pred in predictions], axis=1
                )
                combined_df.columns = train.components
                pred_ts = TimeSeries.from_dataframe(combined_df)
                all_predictions.append(pred_ts)

            # Calculate training time
            training_time = time.time() - start_time
            wandb.log({"training_time": training_time})

            # Log metrics if wandb_logger is available
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
        """Get list of enabled model names"""
        return list(self.models.keys())
