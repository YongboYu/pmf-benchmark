import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import optuna


class OptunaManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create hierarchical directory structure
        self.study_dir = Path("logs/optuna/studies")
        self.study_dir.mkdir(parents=True, exist_ok=True)

    def create_study(
        self, dataset: str, horizon: int, model_group: str, model_name: str
    ) -> optuna.Study:
        """Create or load an Optuna study with hierarchical structure"""
        try:
            # Create study path
            study_path = self.study_dir / dataset / f"h{horizon}" / model_group
            study_path.mkdir(parents=True, exist_ok=True)

            # Add timestamp to make each study unique
            timestamp = int(time.time())
            study_name = f"{dataset}_h{horizon}_{model_group}_{model_name}_{timestamp}"

            # Use SQLite storage with absolute path
            storage_path = study_path / f"{model_name}_{timestamp}.db"
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_name = f"sqlite:///{storage_path.absolute()}"

            # Create new study with proper exception handling
            try:
                study = optuna.create_study(
                    study_name=study_name,
                    storage=storage_name,
                    direction="minimize",
                    load_if_exists=True,  # Changed to True to handle concurrent creation
                )
            except Exception as storage_error:
                self.logger.error(f"Error with SQLite storage: {storage_error}")
                # Fallback to in-memory storage
                self.logger.warning("Falling back to in-memory storage")
                study = optuna.create_study(study_name=study_name, direction="minimize")

            self.logger.info(f"Created new study: {study_name}")
            return study

        except Exception as e:
            self.logger.error(f"Error creating study: {e}")
            return optuna.create_study(
                study_name=f"{model_name}_{int(time.time())}", direction="minimize"
            )

    def optimize(
        self,
        study: optuna.Study,
        objective: Callable,
        n_trials: int,
        timeout: int | None = None,
    ) -> optuna.Study:
        """Run optimization"""
        try:
            # Wrap the objective function to handle exceptions properly
            def wrapped_objective(trial: optuna.Trial) -> float:
                try:
                    return objective(trial)
                except Exception as e:
                    self.logger.error(f"Trial failed: {e}")
                    raise optuna.exceptions.TrialPruned() from e

            # Run optimization with proper exception handling
            study.optimize(
                wrapped_objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
                catch=(
                    Exception,
                ),  # Catch all exceptions to prevent study from getting stuck
            )
            return study
        except Exception as e:
            self.logger.error(f"Error during optimization: {e}")
            raise

    def save_study(
        self,
        study: optuna.Study,
        dataset: str,
        horizon: int,
        model_group: str,
        model_name: str,
    ):
        """Save study results in hierarchical structure"""
        try:
            study_path = (
                self.study_dir
                / dataset
                / f"h{horizon}"
                / model_group
                / f"{model_name}_study.pkl"
            )
            study_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(study, study_path)
            self.logger.info(f"Successfully saved study to {study_path}")
        except Exception as e:
            self.logger.error(f"Error saving study: {e}")

    def load_study(
        self, dataset: str, horizon: int, model_group: str, model_name: str
    ) -> optuna.Study | None:
        """Load existing study from hierarchical structure"""
        try:
            study_path = (
                self.study_dir
                / dataset
                / f"h{horizon}"
                / model_group
                / f"{model_name}_study.pkl"
            )
            if study_path.exists():
                return joblib.load(study_path)
        except Exception as e:
            self.logger.error(f"Error loading study: {e}")
        return None
