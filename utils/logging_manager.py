import logging
import os
from pathlib import Path
from typing import Any

import optuna
import wandb


class LoggingManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logs_dir = Path("logs")
        self.wandb_dir = self.logs_dir / "wandb"
        self.optuna_dir = self.logs_dir / "optuna"
        self.data_preprocess_dir = self.logs_dir / "data_preprocess"
        self.error_logs_dir = self.logs_dir / "error_logs"

        # Create all logging directories
        for dir_path in [
            self.logs_dir,
            self.wandb_dir,
            self.optuna_dir,
            self.data_preprocess_dir,
            self.error_logs_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Setup basic logging
        self._setup_file_logging()

        # Set wandb directory
        os.environ["WANDB_DIR"] = str(self.wandb_dir)

    def _setup_file_logging(self):
        """Configure basic file logging"""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.logs_dir / "training.log"),
                logging.StreamHandler(),
            ],
        )

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name"""
        logger = logging.getLogger(name)
        return logger

    def get_preprocessing_logger(self, dataset: str) -> logging.Logger:
        """Get logger for preprocessing specific dataset"""
        logger = logging.getLogger(f"preprocessing.{dataset}")

        # Add file handler for dataset-specific logs
        fh = logging.FileHandler(self.data_preprocess_dir / f"{dataset}.txt")
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)

        return logger

    def setup_wandb(self, run_config: dict[str, Any], run_name: str | None = None):
        """Initialize WandB run"""
        return wandb.init(
            project=self.config["project"]["wandb_project"],
            name=run_name,
            dir=str(self.wandb_dir),
            config=run_config,
        )

    def setup_optuna_study(
        self, model_name: str, dataset: str, horizon: int
    ) -> optuna.Study:
        """Initialize Optuna study"""
        study_name = f"{model_name}_{dataset}_h{horizon}"
        db_path = self.optuna_dir / "studies" / f"{study_name}.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)

        return optuna.create_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path}",
            direction="minimize",
            load_if_exists=True,
        )


# Singleton pattern for LoggingManager
def get_logging_manager(config: dict[str, Any]) -> LoggingManager:
    if not hasattr(get_logging_manager, "_instance"):
        get_logging_manager._instance = LoggingManager(config)
    return get_logging_manager._instance
