import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, rmse

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.results_dir = Path("results")
        self.temp_results_dir = self.results_dir / "temp"
        self.temp_results_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_predictions(
        self, predictions: TimeSeries, actuals: TimeSeries
    ) -> dict[str, float]:
        """Evaluate predictions against actuals on their current scale"""
        return {"mae": mae(actuals, predictions), "rmse": rmse(actuals, predictions)}

    def evaluate_model_group(
        self,
        predictions: dict[str, TimeSeries],
        test: TimeSeries,
        transformer: object | None = None,
    ) -> dict[str, dict[str, float]]:
        """Evaluate all models in a group"""
        return {
            model_name: self.evaluate_predictions(pred, test)
            for model_name, pred in predictions.items()
        }

    def get_results_paths(
        self, dataset: str, horizon: int, model_group: str
    ) -> dict[str, Path]:
        """Get all relevant paths for saving results with the new structure

        Args:
            dataset: Name of the dataset
            horizon: Forecast horizon
            model_group: Model group name

        Returns:
            Dictionary containing paths for evaluation, models, predictions
        """
        base_path = self.results_dir / dataset / f"horizon_{horizon}"

        paths = {
            "evaluation": base_path / "evaluation" / model_group,
            "models": base_path / "models" / model_group,
            "predictions": base_path / "predictions" / model_group,
            "temp": self.results_dir / dataset / "temp",  # Keep temp at dataset level
        }

        # Create directories if they don't exist
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)

        return paths

    def save_model(
        self, model: Any, dataset: str, model_group: str, model_name: str, horizon: int
    ) -> str | None:
        """Save trained model"""
        try:
            paths = self.get_results_paths(dataset, horizon, model_group)
            model_name_str = str(model_name).replace("/", "_")

            # Handle univariate models (both regression and dl)
            if model_group in [
                "univariate_regression",
                "covariate_regression",
                "univariate_dl",
                "covariate_dl",
            ] and isinstance(model, list):
                saved_paths = []
                for idx, component_model in enumerate(model):
                    component_path = (
                        paths["models"] / f"{model_name_str}_component_{idx}.pkl"
                    )
                    component_path.parent.mkdir(parents=True, exist_ok=True)

                    if hasattr(component_model, "save"):
                        component_model.save(str(component_path))
                        saved_paths.append(str(component_path))
                return saved_paths[0] if saved_paths else None

            # Regular model saving logic
            save_path = paths["models"] / model_name_str
            if not save_path.suffix:
                save_path = save_path.with_suffix(".pkl")

            save_path.parent.mkdir(parents=True, exist_ok=True)

            if hasattr(model, "save"):
                model.save(str(save_path))
                return str(save_path)
            else:
                logger.warning(f"Model {model_name} does not support saving")
                return None

        except Exception as e:
            logger.error(f"Error saving model {model_name}: {e!s}")
            return None

    def save_predictions(
        self,
        predictions: list[TimeSeries],
        dataset: str,
        model_group: str,
        model_name: str,
        horizon: int,
    ) -> dict[str, str]:
        """Save both all-points and last-point predictions in Parquet format"""
        paths = self.get_results_paths(dataset, horizon, model_group)

        # Save all-points predictions
        all_points_df = self.transform_predictions_to_dataframe(
            predictions, last_only=False
        )
        all_points_path = paths["predictions"] / f"{model_name}_all_predictions.parquet"
        all_points_df.to_parquet(all_points_path, engine="pyarrow", index=True)

        # Save last-point predictions
        last_point_df = self.transform_predictions_to_dataframe(
            predictions, last_only=True
        )
        last_point_path = (
            paths["predictions"] / f"{model_name}_last_predictions.parquet"
        )
        last_point_df.to_parquet(last_point_path, engine="pyarrow", index=True)

        return {"all_points": str(all_points_path), "last_point": str(last_point_path)}

    def save_metrics(
        self,
        metrics_df: pd.DataFrame,
        dataset: str,
        model_group: str,
        model_name: str,
        horizon: int,
    ) -> str:
        """Save both all-points and last-point metrics in a single CSV file"""
        paths = self.get_results_paths(dataset, horizon, model_group)

        # Create a single row with all metrics
        combined_metrics = pd.DataFrame(
            [
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "model": f"{model_group}_{model_name}",
                    "all_points_mae": metrics_df["all_points"]["mae"],
                    "all_points_rmse": metrics_df["all_points"]["rmse"],
                    "last_point_mae": metrics_df["last_point"]["mae"],
                    "last_point_rmse": metrics_df["last_point"]["rmse"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            ]
        )

        # Save metrics
        metrics_path = paths["evaluation"] / f"{model_name}_combined_metrics.csv"

        # If file exists, append; otherwise create new
        if metrics_path.exists():
            existing_metrics = pd.read_csv(metrics_path)
            combined_metrics = pd.concat(
                [existing_metrics, combined_metrics], ignore_index=True
            )

        combined_metrics.to_csv(metrics_path, index=False)

        return str(metrics_path)

    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def save_all_results(
        self,
        model_results: dict[str, Any],
        dataset: str,
        horizon: int,
        predictions: list[TimeSeries],
    ) -> dict[str, Any]:
        """Save all model results including model, predictions, metrics, and temporary results

        Args:
            model_results: Dictionary containing model results and metrics
            dataset: Name of the dataset
            horizon: Forecast horizon
            predictions: List of TimeSeries predictions

        Returns:
            Dictionary containing all saved paths
        """
        saved_paths = {}
        model_group = model_results["model_name"]
        model_name = model_results["specific_model"]

        # Save trained model if provided and enabled
        if model_results.get("model") and self.config["evaluation"]["save_models"]:
            # Handle both statistical and univariate regression models
            if isinstance(model_results["model"], list):
                saved_model_paths = []
                for idx, component_model in enumerate(model_results["model"]):
                    model_path = self.save_model(
                        model=component_model,
                        dataset=dataset,
                        model_group=model_group,
                        model_name=f"{model_name}/component_{idx}",
                        horizon=horizon,
                    )
                    if model_path:
                        saved_model_paths.append(model_path)
                if saved_model_paths:
                    saved_paths["model_paths"] = saved_model_paths
            else:
                model_path = self.save_model(
                    model=model_results["model"],
                    dataset=dataset,
                    model_group=model_group,
                    model_name=model_name,
                    horizon=horizon,
                )
                if model_path:
                    saved_paths["model_path"] = model_path

        # Save predictions if enabled
        if predictions and self.config["evaluation"]["save_predictions"]:
            pred_paths = self.save_predictions(
                predictions=predictions,
                dataset=dataset,
                model_group=model_group,
                model_name=model_name,
                horizon=horizon,
            )
            saved_paths["predictions"] = pred_paths

        # Save metrics
        if "metrics" in model_results:
            metrics_path = self.save_metrics(
                metrics_df=model_results["metrics"],
                dataset=dataset,
                model_group=model_group,
                model_name=model_name,
                horizon=horizon,
            )
            saved_paths["metrics"] = metrics_path

        # Save temporary results for later merging
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        temp_result = {
            "dataset": dataset,
            "horizon": horizon,
            "model_group": model_group,
            "model_name": model_name,
            "metrics": self._convert_numpy_types(model_results["metrics"]),
            "training_time": float(model_results.get("training_time", 0)),
            "timestamp": timestamp,
        }

        temp_file = (
            self.temp_results_dir
            / f"{dataset}_{horizon}_{model_group}_{model_name}.json"
        )
        with open(temp_file, "w") as f:
            json.dump(temp_result, f, indent=4)

        saved_paths["temp_result"] = str(temp_file)

        return saved_paths

    def merge_results(self):
        """Merge all temporary result files into final results_log.json"""
        final_results = {
            "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "datasets": {},
        }

        # Read all temporary files
        for temp_file in self.temp_results_dir.glob("*.json"):
            with open(temp_file) as f:
                result = json.load(f)

            dataset = result["dataset"]
            horizon = str(result["horizon"])
            model_group = result["model_group"]
            model_name = result["model_name"]

            # Build nested structure for datasets
            if dataset not in final_results["datasets"]:
                final_results["datasets"][dataset] = {
                    "last_updated": result["timestamp"],
                    "horizons": {},
                }

            if horizon not in final_results["datasets"][dataset]["horizons"]:
                final_results["datasets"][dataset]["horizons"][horizon] = {
                    "last_updated": result["timestamp"],
                    "model_groups": {},
                }

            if (
                model_group
                not in final_results["datasets"][dataset]["horizons"][horizon][
                    "model_groups"
                ]
            ):
                final_results["datasets"][dataset]["horizons"][horizon]["model_groups"][
                    model_group
                ] = {"last_updated": result["timestamp"], "models": {}}

            # Add model results
            final_results["datasets"][dataset]["horizons"][horizon]["model_groups"][
                model_group
            ]["models"][model_name] = {
                "metrics": result["metrics"],
                "training_time": result["training_time"],
                "timestamp": result["timestamp"],
            }

        # Save merged results
        results_file = self.results_dir / "results_log.json"
        with open(results_file, "w") as f:
            json.dump(final_results, f, indent=4)

        # Cleanup temporary files
        # for temp_file in self.temp_results_dir.glob("*.json"):
        #     temp_file.unlink()

    def load_model(
        self, dataset: str, model_group: str, model_name: str, horizon: int
    ) -> Any | None:
        """Load a saved model from disk"""
        try:
            paths = self.get_results_paths(dataset, horizon, model_group)

            # Handle component-based models (univariate/covariate models)
            if model_group in [
                "univariate_regression",
                "covariate_regression",
                "univariate_dl",
                "covariate_dl",
            ]:
                models = []
                base_path = paths["models"]
                model_files = list(base_path.glob(f"{model_name}_component_*.pkl"))

                if not model_files:
                    logger.warning(f"No component models found for {model_name}")
                    return None

                for model_file in sorted(model_files):
                    from darts.models import load_model

                    component_model = load_model(model_file)
                    models.append(component_model)
                return models

            # Regular model loading logic
            load_path = paths["models"] / f"{model_name}.pkl"

            if not load_path.exists():
                logger.warning(f"No saved model found at {load_path}")
                return None

            # For Darts models
            if model_group in [
                "deep_learning",
                "regression",
                "univariate_regression",
                "covariate_regression",
                "univariate_dl",
                "covariate_dl",
            ]:
                from darts.models import load_model

                return load_model(load_path)

            # For other models (using joblib)
            import joblib

            return joblib.load(load_path)

        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e!s}")
            return None

    def evaluate_sequence_predictions(
        self, predictions: list[TimeSeries], actuals: list[TimeSeries]
    ) -> dict[str, dict[str, Any]]:
        """Evaluate sequence predictions using both all-points and last-point methods

        Args:
            predictions: List of prediction TimeSeries
            actuals: List of actual TimeSeries

        Returns:
            Dictionary containing:
            - Overall all-points and last-point metrics
            - Per-component all-points and last-point metrics
        """
        # Initialize metrics containers
        all_points_metrics = []
        last_point_metrics = []

        # Initialize per-component metrics
        components = actuals[0].components
        per_component_metrics = {
            component: {
                "all_points": {"mae": [], "rmse": []},
                "last_point": {"mae": [], "rmse": []},
            }
            for component in components
        }

        # 1. All-points evaluation
        for pred, true in zip(predictions, actuals, strict=False):
            # Overall metrics
            metrics = self.evaluate_predictions(pred, true)
            all_points_metrics.append(metrics)

            # Per-component metrics
            for component in components:
                pred_component = pred[component]
                true_component = true[component]
                component_metrics = self.evaluate_predictions(
                    pred_component, true_component
                )
                per_component_metrics[component]["all_points"]["mae"].append(
                    component_metrics["mae"]
                )
                per_component_metrics[component]["all_points"]["rmse"].append(
                    component_metrics["rmse"]
                )

        # 2. Last-point evaluation
        for pred, true in zip(predictions, actuals, strict=False):
            # Overall metrics
            last_pred = TimeSeries.from_values(pred.values()[-1])
            last_true = TimeSeries.from_values(true.values()[-1])
            metrics = self.evaluate_predictions(last_pred, last_true)
            last_point_metrics.append(metrics)

            # Per-component metrics
            for component in components:
                last_pred_component = TimeSeries.from_values(
                    pred[component].values()[-1]
                )
                last_true_component = TimeSeries.from_values(
                    true[component].values()[-1]
                )
                component_metrics = self.evaluate_predictions(
                    last_pred_component, last_true_component
                )
                per_component_metrics[component]["last_point"]["mae"].append(
                    component_metrics["mae"]
                )
                per_component_metrics[component]["last_point"]["rmse"].append(
                    component_metrics["rmse"]
                )

        # Calculate average overall metrics
        avg_all_points = {
            metric: np.mean([m[metric] for m in all_points_metrics])
            for metric in ["mae", "rmse"]
        }

        avg_last_point = {
            metric: np.mean([m[metric] for m in last_point_metrics])
            for metric in ["mae", "rmse"]
        }

        # Calculate average per-component metrics
        for component in components:
            for metric_type in ["all_points", "last_point"]:
                for metric in ["mae", "rmse"]:
                    values = per_component_metrics[component][metric_type][metric]
                    per_component_metrics[component][metric_type][metric] = np.mean(
                        values
                    )

        return {
            "overall": {"all_points": avg_all_points, "last_point": avg_last_point},
            "per_component": per_component_metrics,
        }

    def transform_predictions_to_dataframe(
        self, predictions: list[TimeSeries], last_only: bool = False
    ) -> pd.DataFrame:
        """Transform predictions into a structured DataFrame format.

        Args:
            predictions: List of TimeSeries predictions
            last_only: If True, only return the last point of each sequence

        Returns:
            DataFrame with:
            - If last_only=False: 3D structure (sequence_start_time x horizon x variables)
            - If last_only=True: 2D structure (sequence_start_time x variables)
        """
        if not predictions:
            raise ValueError("Empty predictions list")

        # Get basic information
        horizon = len(predictions[0])
        n_components = predictions[0].width
        components = predictions[0].components

        if last_only:
            # Create 2D DataFrame for last points only
            data = {
                components[i]: [pred.values()[-1][i] for pred in predictions]
                for i in range(n_components)
            }

            # Create index using the sequence start time + horizon
            index = [
                pred.start_time() + pd.Timedelta(days=horizon - 1)
                for pred in predictions
            ]

            df = pd.DataFrame(data, index=index)
            df.index.name = "time"

        else:
            # Create 3D DataFrame using MultiIndex
            rows = []

            for _seq_idx, pred in enumerate(predictions):
                sequence_start = pred.start_time()

                for step in range(horizon):
                    row = {
                        "sequence_start_time": sequence_start,
                        "horizon_step": step,
                    }

                    # Add values for each component
                    for comp_idx in range(n_components):
                        row[components[comp_idx]] = pred.values()[step][comp_idx]

                    rows.append(row)

            # Create DataFrame with MultiIndex
            df = pd.DataFrame(rows)
            df = df.set_index(["sequence_start_time", "horizon_step"])

        return df

    def generate_results_csv(self, json_path: str | Path, output_path: str | Path):
        """Convert results JSON to a flattened CSV format

        Args:
            json_path: Path to the results JSON file
            output_path: Path to save the CSV file
        """
        # Convert string paths to Path objects if needed
        json_path = Path(json_path)
        output_path = Path(output_path)

        # Read JSON file
        with open(json_path) as f:
            results = json.load(f)

        # Initialize list to store flattened records
        records = []

        # Iterate through the nested structure
        for dataset, dataset_data in results["datasets"].items():
            for horizon, horizon_data in dataset_data["horizons"].items():
                for group_name, group_data in horizon_data["model_groups"].items():
                    for model_name, model_data in group_data["models"].items():
                        record = {
                            "dataset": dataset,
                            "horizon": horizon,
                            "model_group": group_name,
                            "model": model_name,
                            "timestamp": model_data["timestamp"],
                            "training_time": model_data["training_time"],
                        }

                        # Add all metrics (both all_points and last_point)
                        for metric_type, metric_values in model_data["metrics"].items():
                            for metric_name, value in metric_values.items():
                                record[f"{metric_type}_{metric_name}"] = value

                        records.append(record)

        # Convert to DataFrame and save as CSV
        df = pd.DataFrame(records)

        # Reorder columns for better readability
        base_columns = [
            "dataset",
            "horizon",
            "model_group",
            "model",
            "timestamp",
            "training_time",
        ]
        metric_columns = [col for col in df.columns if col not in base_columns]
        df = df[base_columns + sorted(metric_columns)]

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        df.to_csv(output_path, index=False)

        return df

    def evaluate_top_frequent_components(
        self,
        predictions: list[TimeSeries],
        actuals: list[TimeSeries],
        train: TimeSeries,
        dataset: str,
        model_group: str,
        model_name: str,
        horizon: int,
        n_components: int = 10,
    ) -> dict[str, Any]:
        """Evaluate predictions for top N most frequent time series components based on training data

        Args:
            predictions: List of prediction TimeSeries
            actuals: List of actual TimeSeries
            train: Training TimeSeries used to determine frequency
            dataset: Name of the dataset
            model_group: Name of the model group
            model_name: Name of the specific model
            horizon: Forecast horizon
            n_components: Number of top components to evaluate (default: 10)

        Returns:
            Dictionary containing:
            - List of top component names
            - Overall metrics for top components
            - Per-component metrics for top components
        """
        # Get the top N most frequent components based on training data
        top_components = self._get_top_frequent_components(train, n=n_components)

        # Save the list of top components
        self._save_top_components_list(top_components, dataset, horizon)

        # Initialize per-component metrics
        per_component_metrics = {
            component: {
                "all_points": {"mae": [], "rmse": []},
                "last_point": {"mae": [], "rmse": []},
            }
            for component in top_components
        }

        # 1. All-points evaluation for top components
        for pred, true in zip(predictions, actuals, strict=False):
            # Only evaluate top components
            for component in top_components:
                if component in pred.components:
                    pred_component = pred[component]
                    true_component = true[component]
                    component_metrics = self.evaluate_predictions(
                        pred_component, true_component
                    )
                    per_component_metrics[component]["all_points"]["mae"].append(
                        component_metrics["mae"]
                    )
                    per_component_metrics[component]["all_points"]["rmse"].append(
                        component_metrics["rmse"]
                    )

        # 2. Last-point evaluation for top components
        for pred, true in zip(predictions, actuals, strict=False):
            # Only evaluate top components
            for component in top_components:
                if component in pred.components:
                    last_pred_component = TimeSeries.from_values(
                        pred[component].values()[-1]
                    )
                    last_true_component = TimeSeries.from_values(
                        true[component].values()[-1]
                    )
                    component_metrics = self.evaluate_predictions(
                        last_pred_component, last_true_component
                    )
                    per_component_metrics[component]["last_point"]["mae"].append(
                        component_metrics["mae"]
                    )
                    per_component_metrics[component]["last_point"]["rmse"].append(
                        component_metrics["rmse"]
                    )

        # Calculate average per-component metrics
        for component in top_components:
            for metric_type in ["all_points", "last_point"]:
                for metric in ["mae", "rmse"]:
                    values = per_component_metrics[component][metric_type][metric]
                    if values:  # Check if component was evaluated
                        per_component_metrics[component][metric_type][metric] = np.mean(
                            values
                        )

        # Calculate average overall metrics across top components
        avg_all_points = {
            metric: np.mean(
                [
                    m["all_points"][metric]
                    for m in per_component_metrics.values()
                    if "all_points" in m
                    and metric in m["all_points"]
                    and isinstance(m["all_points"][metric], (int, float))
                ]
            )
            for metric in ["mae", "rmse"]
        }

        avg_last_point = {
            metric: np.mean(
                [
                    m["last_point"][metric]
                    for m in per_component_metrics.values()
                    if "last_point" in m
                    and metric in m["last_point"]
                    and isinstance(m["last_point"][metric], (int, float))
                ]
            )
            for metric in ["mae", "rmse"]
        }

        # Prepare results dictionary
        results = {
            "top_components": top_components,
            "overall": {"all_points": avg_all_points, "last_point": avg_last_point},
            "per_component": per_component_metrics,
        }

        # Save the evaluation results
        self._save_top_components_evaluation(
            results, dataset, model_group, model_name, horizon
        )

        return results

    def _get_top_frequent_components(self, train: TimeSeries, n: int = 10) -> list[str]:
        """Get top N most frequent components based on training data means

        Args:
            train: Training TimeSeries data
            n: Number of top components to return

        Returns:
            List of component names sorted by frequency
        """
        # Calculate mean value for each component
        means = {
            component: np.mean(train[component].values().flatten())
            for component in train.components
        }

        # Sort components by mean value and get top N
        top_components = sorted(means.items(), key=lambda x: x[1], reverse=True)[:n]

        logger.info("\nTop frequent components:")
        for comp, mean_val in top_components:
            logger.info(f"{comp}: {mean_val:.4f}")

        return [comp[0] for comp in top_components]

    def _save_top_components_list(
        self, top_components: list[str], dataset: str, horizon: int
    ):
        """Save list of top components to disk

        Args:
            top_components: List of top component names
            dataset: Name of the dataset
            horizon: Forecast horizon
        """
        base_path = self.results_dir / dataset / f"horizon_{horizon}"
        top_comp_dir = base_path / "top_components"
        top_comp_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        top_comp_path = top_comp_dir / "top_components.json"
        with open(top_comp_path, "w") as f:
            json.dump(
                {
                    "dataset": dataset,
                    "horizon": horizon,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "top_components": top_components,
                },
                f,
                indent=4,
            )

        logger.info(f"Saved top components list to {top_comp_path}")

    def _save_top_components_evaluation(
        self,
        results: dict[str, Any],
        dataset: str,
        model_group: str,
        model_name: str,
        horizon: int,
    ):
        """Save evaluation results for top components

        Args:
            results: Evaluation results dictionary
            dataset: Name of the dataset
            model_group: Name of the model group
            model_name: Name of the specific model
            horizon: Forecast horizon
        """
        base_path = self.results_dir / dataset / f"horizon_{horizon}"
        top_eval_dir = base_path / "top_evaluation" / model_group
        top_eval_dir.mkdir(parents=True, exist_ok=True)

        # Create a single row with metrics
        metrics_row = {
            "dataset": dataset,
            "horizon": horizon,
            "model": f"{model_group}_{model_name}",
            "top_all_points_mae": results["overall"]["all_points"]["mae"],
            "top_all_points_rmse": results["overall"]["all_points"]["rmse"],
            "top_last_point_mae": results["overall"]["last_point"]["mae"],
            "top_last_point_rmse": results["overall"]["last_point"]["rmse"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Add per-component metrics
        for comp in results["top_components"]:
            metrics_row[f"{comp}_all_points_mae"] = results["per_component"][comp][
                "all_points"
            ]["mae"]
            metrics_row[f"{comp}_all_points_rmse"] = results["per_component"][comp][
                "all_points"
            ]["rmse"]
            metrics_row[f"{comp}_last_point_mae"] = results["per_component"][comp][
                "last_point"
            ]["mae"]
            metrics_row[f"{comp}_last_point_rmse"] = results["per_component"][comp][
                "last_point"
            ]["rmse"]

        # Convert to DataFrame
        metrics_df = pd.DataFrame([metrics_row])

        # Save metrics
        metrics_path = top_eval_dir / f"{model_name}_top_metrics.csv"

        # If file exists, append; otherwise create new
        if metrics_path.exists():
            existing_metrics = pd.read_csv(metrics_path)
            metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)

        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"Saved top components evaluation to {metrics_path}")


#######################################


def merge_result_csvs(results_dir: str | Path) -> pd.DataFrame:
    """Merge metrics CSV files from evaluation folders under results/dataset/horizon/.

    Expected structure:
    results/
        dataset1/
            horizon1/
                evaluation/
                    metrics.csv
            horizon2/
                evaluation/
                    metrics.csv
        dataset2/
            ...

    Args:
        results_dir: Root results directory containing dataset folders

    Returns:
        pd.DataFrame: Merged metrics from all CSV files
    """
    results_dir = Path(results_dir)

    # Find all evaluation folders
    eval_paths = list(results_dir.glob("*/*/evaluation/*"))

    if not eval_paths:
        raise FileNotFoundError(f"No evaluation folders found in {results_dir}")

    dfs = []
    for eval_path in eval_paths:
        # Extract dataset and horizon from path
        dataset = eval_path.parent.parent.name
        horizon = eval_path.parent.name

        # Find CSV files in evaluation folder
        csv_files = list(eval_path.glob("*.csv"))

        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            # Add dataset and horizon if not present
            if "dataset" not in df.columns:
                df["dataset"] = dataset
            if "horizon" not in df.columns:
                df["horizon"] = horizon
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No CSV files found in evaluation folders")

    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save merged results
    output_path = results_dir / "merged_metrics.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Merged metrics saved to: {output_path}")

    return merged_df


def merge_top_components_csvs(results_dir: str | Path) -> pd.DataFrame:
    """Merge top components metrics CSV files from top_evaluation folders.

    Expected structure:
    results/
        dataset1/
            horizon1/
                top_evaluation/
                    model_group1/
                        model_name1_top_metrics.csv
            horizon2/
                top_evaluation/
                    model_group1/
                        model_name1_top_metrics.csv
        dataset2/
            ...

    Args:
        results_dir: Root results directory containing dataset folders

    Returns:
        pd.DataFrame: Merged top components metrics from all CSV files
    """
    results_dir = Path(results_dir)

    # Find all top_evaluation folders
    eval_paths = list(results_dir.glob("*/*/top_evaluation/*/*"))

    if not eval_paths:
        raise FileNotFoundError(f"No top evaluation files found in {results_dir}")

    dfs = []
    for eval_path in eval_paths:
        if not eval_path.is_file() or eval_path.suffix != ".csv":
            continue

        # Extract dataset and horizon from path
        dataset = eval_path.parent.parent.parent.parent.name
        horizon = eval_path.parent.parent.parent.name
        model_group = eval_path.parent.name

        # Read CSV file
        df = pd.read_csv(eval_path)

        # Add dataset, horizon, and model_group if not present
        if "dataset" not in df.columns:
            df["dataset"] = dataset
        if "horizon" not in df.columns:
            df["horizon"] = horizon
        if "model_group" not in df.columns and "model" in df.columns:
            # Extract model_group from model column if needed
            df["model_group"] = df["model"].apply(
                lambda x, model_group=model_group: (
                    x.split("_")[0] if "_" in x else model_group
                )
            )

        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No CSV files found in top_evaluation folders")

    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # Save merged results
    output_path = results_dir / "merged_top_components_metrics.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Merged top components metrics saved to: {output_path}")

    return merged_df


def evaluate_entropic_relevance(
    dataset: str,
    model_group: str,
    model_name: str,
    horizon: int,
    forecast_confidence: str = "80",
) -> dict[str, Any]:
    """
    Evaluate model predictions using entropic relevance.

    Args:
        dataset: Name of the dataset
        model_group: Model group name (e.g., 'deep_learning')
        model_name: Model name (e.g., 'deepar')
        horizon: Forecast horizon (e.g., 7)
        forecast_confidence: Confidence level for forecast (80, 90, 95, 100)

    Returns:
        Dictionary containing ER metrics
    """
    import json
    import os
    from pathlib import Path

    import pandas as pd

    # Try to import calculate_entropic_relevance
    try:
        from calculate_entropic_relevance_corrected import (
            calculate_entropic_relevance as calc_er,
        )
    except ImportError:
        logger.error(
            "calculate_entropic_relevance_corrected.py not found. Make sure it's in the path."
        )
        return {"error": "calculate_entropic_relevance_corrected.py not found"}

    # Path to predictions
    prediction_path = Path(
        f"results/{dataset}/horizon_{horizon}/predictions/{model_group}/{model_name}_all_predictions.parquet"
    )

    if not prediction_path.exists():
        logger.error(f"Prediction file not found: {prediction_path}")
        return {"error": f"Prediction file not found: {prediction_path}"}

    # Path to ground truth data
    ground_truth_path = Path(f"data/raw/{dataset}.xes")

    if not ground_truth_path.exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        return {"error": f"Ground truth file not found: {ground_truth_path}"}

    # Import xes importer here to avoid unnecessary dependencies
    try:
        from pm4py.objects.log.importer.xes import importer as xes_importer
    except ImportError:
        logger.error("pm4py not installed. Install it with 'pip install pm4py'")
        return {"error": "pm4py not installed"}

    # Create output directory
    os.makedirs("er_results", exist_ok=True)

    # Load predictions
    predictions_df = pd.read_parquet(prediction_path)

    # Extract directly-follows relations
    reverse_map = {}
    reverse_map["Start"] = 0
    reverse_map["End"] = 1

    # Process each column in predictions_df
    # Each column represents a directly-follows relation
    for df_relation in predictions_df.columns:
        # Some columns might already be using '▶' or '■' as symbols for start/end
        source, end = df_relation.replace("▶", "Start").replace("■", "End").split("->")
        source = source.strip()
        end = end.strip()

        if source not in reverse_map:
            reverse_map[source] = len(reverse_map)
        if end not in reverse_map:
            reverse_map[end] = len(reverse_map)

    # Create DFG structure
    arcs = []
    node_freq = dict.fromkeys(reverse_map.keys(), 0)

    # Process each directly-follows relation
    for df_relation in predictions_df.columns:
        clean_relation = df_relation.replace("▶", "Start").replace("■", "End")
        source, end = [part.strip() for part in clean_relation.split("->")]

        # For each time step in the horizon, extract the prediction value
        for _, row in predictions_df.iterrows():
            freq = round(float(row[df_relation]))
            if freq <= 0:
                continue

            arcs.append(
                {"from": reverse_map[source], "to": reverse_map[end], "freq": freq}
            )

            if source == "Start":
                node_freq[source] += freq
            else:
                node_freq[end] += freq

    # Create nodes
    nodes = []
    for node, freq in node_freq.items():
        nodes.append({"label": node, "id": reverse_map[node], "freq": round(freq)})

    # Create DFG JSON
    dfg = {"nodes": nodes, "arcs": arcs}

    # Save temporary JSON file
    temp_json = f"temp_{dataset}_{horizon}_{model_group}_{model_name}.json"
    with open(temp_json, "w") as f:
        json.dump(dfg, f, indent=1)

    # Load ground truth XES log
    try:
        variant = xes_importer.Variants.ITERPARSE
        parameters = {variant.value.Parameters.MAX_TRACES: 100000}
        log = xes_importer.apply(str(ground_truth_path), parameters=parameters)
    except Exception as e:
        logger.error(f"Error loading XES log: {e}")
        # Clean up temporary JSON file
        if os.path.exists(temp_json):
            os.remove(temp_json)
        return {"error": f"Error loading XES log: {e}"}

    # Calculate entropic relevance
    try:
        er, non_fitting_traces, total_traces = calc_er(temp_json, log, model_name)
        logger.info(f"Entropic Relevance: {er}")

        # Clean up temporary JSON file
        if os.path.exists(temp_json):
            os.remove(temp_json)

        # Write results to CSV
        result_file = Path(f"er_results/{dataset}_er_results.csv")

        # Create header if file doesn't exist
        if not result_file.exists():
            with open(result_file, "w") as f:
                f.write(
                    "dataset,horizon,model_group,model_name,forecast_confidence,er,non_fitting_traces,total_traces\n"
                )

        # Append results
        with open(result_file, "a") as f:
            f.write(
                f"{dataset},{horizon},{model_group},{model_name},{forecast_confidence},{er},{non_fitting_traces},{total_traces}\n"
            )

        return {
            "entropic_relevance": er,
            "non_fitting_traces": non_fitting_traces,
            "total_traces": total_traces,
            "fitting_ratio": (total_traces - non_fitting_traces) / total_traces
            if total_traces > 0
            else 0,
        }

    except Exception as e:
        logger.error(f"Error calculating entropic relevance: {e}")
        # Clean up temporary JSON file
        if os.path.exists(temp_json):
            os.remove(temp_json)
        return {"error": f"Error calculating entropic relevance: {e}"}


# Example usage
if __name__ == "__main__":
    results_dir = Path("results")  # Adjust if needed
    try:
        merged_df = merge_result_csvs(results_dir)
        print("\nFirst few rows of merged data:")
        print(merged_df.head())
        print("\nColumns in merged data:")
        print(merged_df.columns.tolist())

        # Also merge top components metrics
        top_metrics_df = merge_top_components_csvs(results_dir)
        print("\nFirst few rows of merged top components data:")
        print(top_metrics_df.head())
    except FileNotFoundError as e:
        print(f"Error: {e}")
