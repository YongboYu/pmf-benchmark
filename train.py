"""Train time series forecasting models.

Usage:
    Single model: python train.py --dataset dataset_name --model_group statistical --model prophet --horizon 1
    All models in group: python train.py --dataset dataset_name --model_group statistical --horizon 1
    All models: python train.py --dataset dataset_name --model_group all --horizon 1
"""

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from models.baseline_models import BaselineModels
from models.covariate_dl_models import CovariateDLModels
from models.covariate_regression_models import CovariateRegressionModels
from models.deep_learning_models import DeepLearningModels
from models.foundation_models import FoundationModels
from models.regression_models import RegressionModels
from models.statistical_models import StatisticalModels
from models.univariate_dl_models import UnivariateDLModels
from models.univariate_regression_models import UnivariateRegressionModels
from utils.data_loader import DataLoader
from utils.evaluation import Evaluator
from utils.logging_manager import get_logging_manager
from utils.wandb_logger import WandbLogger

logger = logging.getLogger(__name__)


@dataclass
class ModelResults:
    """Container for model evaluation results."""

    metrics: dict[str, float]
    model_name: str
    specific_model: str
    dataset_name: str
    training_time: float
    horizon: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "specific_model": self.specific_model,
            "dataset_name": self.dataset_name,
            "training_time": self.training_time,
            "horizon": self.horizon,
            "metrics": self.metrics,
        }


class ModelTrainer:
    def __init__(self, config: dict[str, Any], logger: logging.Logger):
        self.config = config
        self.data_loader = DataLoader(config)
        self.evaluator = Evaluator(config)
        self.logger = logger
        self.MODEL_GROUPS = {
            "baseline": BaselineModels,
            "statistical": StatisticalModels,
            "regression": RegressionModels,
            "deep_learning": DeepLearningModels,
            "foundation": FoundationModels,
            "univariate_regression": UnivariateRegressionModels,
            "covariate_regression": CovariateRegressionModels,
            "univariate_dl": UnivariateDLModels,
            "covariate_dl": CovariateDLModels,
        }

    def train_single_model(
        self, dataset: str, model_group: str, model_name: str, horizon: int
    ) -> dict[str, Any]:
        """Train and evaluate a single model for a specific horizon."""
        self.logger.info(
            f"Training {model_group}/{model_name} on {dataset} for horizon {horizon}"
        )

        try:
            # Prepare data and initialize components
            train_t, val_t, test_t, transformer = self.data_loader.prepare_data(
                dataset, model_group
            )

            model = self.MODEL_GROUPS[model_group](self.config)
            wandb_logger = WandbLogger(self.config)

            if model_name not in model.get_model_names():
                raise ValueError(f"Model {model_name} not found in {model_group}")

            results = {}

            # Initialize wandb run
            with wandb_logger.init_run(
                dataset=dataset,
                horizon=horizon,
                model_group=model_group,
                model_name=model_name,
                # config=self.config["model_configs"].get(f"{model_group}_models", {})
                config=self.config,
            ):
                try:
                    # Set the wandb_logger instance in the model
                    if hasattr(model, "wandb_logger"):
                        model.wandb_logger = wandb_logger

                    # start_time = time.time()

                    # Train and get predictions
                    model_results = model.train_and_predict(
                        model_name=model_name,
                        train=train_t,
                        val=val_t,
                        test=test_t,
                        transformer=transformer,
                        horizon=horizon,
                        dataset=dataset,
                        wandb_logger=wandb_logger,
                    )

                    # training_time = time.time() - start_time

                    if model_results.get("predictions") and model_results.get(
                        "actuals"
                    ):
                        # Inverse transform predictions if needed
                        predictions = [
                            self.data_loader.inverse_transform(
                                pred, transformer, model_group
                            )
                            for pred in model_results["predictions"]
                        ]
                        actuals = [
                            self.data_loader.inverse_transform(
                                act, transformer, model_group
                            )
                            for act in model_results["actuals"]
                        ]

                        metrics = self.evaluator.evaluate_sequence_predictions(
                            predictions=predictions,
                            actuals=actuals,
                        )

                        wandb_logger.log_predictions(
                            predictions, actuals, model_name, metrics
                        )
                        wandb_logger.log_results(metrics, model_name)

                        # Save results and metrics
                        result_dict = {
                            "model_name": model_group,
                            "specific_model": model_name,
                            "metrics": {
                                "all_points": metrics["overall"]["all_points"],
                                "last_point": metrics["overall"]["last_point"],
                            },
                            "training_time": model_results["training_time"],
                            "model": model_results["model"],
                            "best_params": model_results.get("best_params", {}),
                        }

                        # Save all results using the evaluator
                        saved_paths = self.evaluator.save_all_results(
                            model_results=result_dict,
                            dataset=dataset,
                            horizon=horizon,
                            predictions=predictions,
                        )

                        # Add saved paths to results
                        result_dict.update(saved_paths)

                        # metrics_with_time = {**metrics, 'training_time': training_time}
                        # wandb_logger.log_metrics(metrics_with_time)

                        results[model_name] = result_dict

                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {e!s}")
                    raise

            return results

        except Exception as e:
            self.logger.error(f"Error training {model_group}/{model_name}: {e!s}")
            raise


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get the directory containing the base config
    config_dir = Path(config_path).parent

    # Load and merge model-specific configs
    model_configs = {}
    model_config_dir = config_dir / "model_configs"

    for config_file in model_config_dir.glob("*.yaml"):
        with open(config_file) as f:
            model_configs[config_file.stem] = yaml.safe_load(f)

    # Add model configs to main config
    config["model_configs"] = model_configs

    return config


def main():
    parser = argparse.ArgumentParser(description="Train time series forecasting models")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument(
        "--model_group",
        type=str,
        required=True,
        choices=[
            "baseline",
            "statistical",
            "regression",
            "univariate_regression",
            "covariate_regression",
            "deep_learning",
            "univariate_dl",
            "covariate_dl",
            "foundation",
            "all",
        ],
        help="Model group to train",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        help="Specific model to train within the model group",
    )
    parser.add_argument(
        "--horizon", type=int, required=True, choices=[7, 28], help="Prediction horizon"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Initialize logging manager
    logging_manager = get_logging_manager(config)
    logger = logging_manager.get_logger("training")

    # Initialize trainer with logger
    trainer = ModelTrainer(config, logger)

    try:
        results = {}
        if args.model_group == "all":
            # Train all models in all groups
            for group in trainer.MODEL_GROUPS:
                model_instance = trainer.MODEL_GROUPS[group](config)
                group_results = {}
                for model_name in model_instance.get_model_names():
                    group_results[model_name] = trainer.train_single_model(
                        args.dataset, group, model_name, args.horizon
                    )
                results[group] = group_results
        else:
            # Get model instance for the specified group
            model_instance = trainer.MODEL_GROUPS[args.model_group](config)

            if args.model:
                # Train specific model
                if args.model not in model_instance.get_model_names():
                    raise ValueError(
                        f"Model '{args.model}' not found in {args.model_group} group"
                    )
                results = {
                    args.model_group: {
                        args.model: trainer.train_single_model(
                            args.dataset, args.model_group, args.model, args.horizon
                        )
                    }
                }
            else:
                # Train all models in the group
                group_results = {}
                for model_name in model_instance.get_model_names():
                    group_results[model_name] = trainer.train_single_model(
                        args.dataset, args.model_group, model_name, args.horizon
                    )
                results = {args.model_group: group_results}

        logger.info("Training completed successfully")
        return results

    except Exception as e:
        logger.error(f"Training failed: {e!s}")
        raise


def train_with_retry(max_retries=3, retry_delay=300):
    for attempt in range(max_retries):
        try:
            results = main()
            return results
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(retry_delay)
            else:
                logging.error(f"All attempts failed: {e}")
                raise


if __name__ == "__main__":
    train_with_retry()
