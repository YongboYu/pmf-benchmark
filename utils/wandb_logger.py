from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import wandb
from darts import TimeSeries
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from utils.evaluation import Evaluator


class WandbLogger:
    def __init__(self, config: dict[str, Any]):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trial_metrics = {}
        self.param_history = {}
        self.trial_history = {}
        self._run = None
        self.config = config

    @property
    def run(self):
        return self._run

    def init_run(
        self,
        dataset: str,
        horizon: int,
        model_group: str,
        model_name: str,
        config: dict[str, Any],
    ):
        """Initialize wandb run with consistent structure"""
        self._run = wandb.init(
            project=f"{config['project']['name']}_{dataset}",
            entity=config["project"]["entity"],
            name=model_name,
            group=model_group,
            job_type=f"horizon_{horizon}",
            # tags=[model_group, model_name],
            # config=config,
            reinit=True,
            dir=str(self.log_dir),
        )
        return self._run

    def log_results(self, metrics: dict[str, Any], model_name: str):
        """Log average metrics for all points and last points

        Args:
            metrics: Dictionary containing overall and per-component metrics
            model_name: Name of the model
        """
        # Extract overall metrics
        overall_metrics = metrics["overall"]

        # Log average metrics
        wandb.log(
            {
                "average_all_mae": overall_metrics["all_points"]["mae"],
                "average_all_rmse": overall_metrics["all_points"]["rmse"],
                "average_last_mae": overall_metrics["last_point"]["mae"],
                "average_last_rmse": overall_metrics["last_point"]["rmse"],
            }
        )

        # Create table data for component metrics
        table_data = []
        columns = ["component", "all_mae", "all_rmse", "last_mae", "last_rmse"]

        for component, component_metrics in metrics["per_component"].items():
            row = [
                component,
                component_metrics["all_points"]["mae"],
                component_metrics["all_points"]["rmse"],
                component_metrics["last_point"]["mae"],
                component_metrics["last_point"]["rmse"],
            ]
            table_data.append(row)

        # Create and log wandb Table
        component_metrics_table = wandb.Table(data=table_data, columns=columns)

        wandb.log({"component_metrics": component_metrics_table})

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | float | None = None,
        prefix: str | None = None,
    ):
        """Log metrics with optional prefix"""
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # If step is a float, convert it to None to avoid the error
        if isinstance(step, float):
            step = None

        wandb.log(metrics, step=step)

    def log_model_artifacts(self, model_name: str, artifacts: dict[str, Any]):
        """Log model-related artifacts"""
        wandb.log(
            {
                f"{model_name}/best_params": artifacts.get("best_params", {}),
                f"{model_name}/best_value": artifacts.get("best_value"),
                f"{model_name}/n_trials": artifacts.get("n_trials", 0),
            }
        )

    def log_training_progress(
        self,
        model_name: str,
        metrics: dict[str, float],
        epoch: int,
        is_validation: bool = False,
    ):
        """Log training progress metrics"""
        prefix = f"{model_name}/{'val' if is_validation else 'train'}"
        self.log_metrics(metrics, step=epoch, prefix=prefix)

    def log_optuna_visualizations(self, study: optuna.Study, model_name: str):
        """Log Optuna study visualizations"""
        try:
            # Optimization history
            history_fig = plot_optimization_history(study)
            wandb.log({f"{model_name}/optimization_history": wandb.Plot(history_fig)})

            if len(study.trials) > 1:
                # Parameter importance
                importance_fig = plot_param_importances(study)
                wandb.log(
                    {f"{model_name}/parameter_importance": wandb.Plot(importance_fig)}
                )

                # Parallel coordinates
                parallel_fig = plot_parallel_coordinate(study)
                wandb.log(
                    {f"{model_name}/parallel_coordinates": wandb.Plot(parallel_fig)}
                )

        except Exception as e:
            wandb.log({f"{model_name}/visualization_error": str(e)})

    def finish(self):
        """Finish the current wandb run"""
        if self._run is not None:
            self._run.finish()
            self._run = None

    def log_predictions(
        self,
        predictions: list[TimeSeries],
        actual: list[TimeSeries],
        model_name: str,
        metrics: dict[str, Any],
    ):
        """Log prediction plots for sequence predictions, showing the last point of each sequence

        Args:
            predictions: List of TimeSeries predictions
            actual: List of ground truth TimeSeries
            model_name: Name of the model
            metrics: Dictionary containing overall and per-component metrics
        """
        # Create evaluator instance with the same config
        evaluator = Evaluator(self.config)

        # Transform predictions and actuals to DataFrames containing last points
        last_predictions_df = evaluator.transform_predictions_to_dataframe(
            predictions, last_only=True
        )
        last_actuals_df = evaluator.transform_predictions_to_dataframe(
            actual, last_only=True
        )

        # Convert DataFrames to numpy arrays for plotting
        last_point_preds = last_predictions_df.values
        last_point_actuals = last_actuals_df.values

        # Get time points from DataFrame index
        time_points = pd.to_datetime(last_predictions_df.index)

        # Create subplot for each time series component
        n_components = actual[0].width
        fig, axes = plt.subplots(n_components, 1, figsize=(12, 4 * n_components))
        if n_components == 1:
            axes = [axes]

        fig.suptitle(f"Model: {model_name}", y=1.02, fontsize=14)

        for i in range(n_components):
            ax = axes[i]
            time_series_name = actual[0].columns[i]

            # Get component metrics
            component_metrics = metrics["per_component"][time_series_name]
            subtitle = f"{time_series_name} (last mae: {component_metrics['last_point']['mae']:.3f}, last rmse: {component_metrics['last_point']['rmse']:.3f})"
            # subtitle = (f"{time_series_name}: "
            #            f"all_mae - {component_metrics['all_points']['mae']:.3f}, "
            #            f"all_rmse - {component_metrics['all_points']['rmse']:.3f}, "
            #            f"last_mae - {component_metrics['last_point']['mae']:.3f}, "
            #            f"last_rmse - {component_metrics['last_point']['rmse']:.3f}")

            # Plot actual values
            ax.plot(
                time_points,
                last_point_actuals[:, i],
                "b-o",
                label="Actual",
                linewidth=2,
                markersize=6,
            )

            # Plot predictions
            ax.plot(
                time_points,
                last_point_preds[:, i],
                "r-o",
                label="Forecast",
                linewidth=2,
                markersize=6,
            )

            # Add error bands
            error = np.abs(last_point_preds[:, i] - last_point_actuals[:, i])
            ax.fill_between(
                time_points,
                last_point_actuals[:, i] - error,
                last_point_actuals[:, i] + error,
                alpha=0.2,
                color="gray",
                label="Error Band",
            )

            # Customize plot
            ax.set_title(subtitle, pad=10)
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Adjust layout
        plt.tight_layout()

        # Log to wandb
        wandb.log(
            {
                f"{model_name}": wandb.Image(fig),
                # f"{model_name}_prediction_stats": {
                #     "overall_metrics": metrics['overall'],
                #     "per_component_metrics": metrics['per_component']
                # }
            }
        )

        plt.close(fig)

    def _log_sequence_plots(
        self, predictions: list[TimeSeries], actual: list[TimeSeries], model_name: str
    ):
        """Log detailed sequence plots for the first few sequences"""
        n_sequences = len(predictions)
        n_components = actual[0].width

        # Create subplot grid
        n_cols = n_components
        n_rows = n_sequences

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Plot each sequence
        for seq_idx in range(n_sequences):
            for comp_idx in range(n_components):
                ax = axes[seq_idx, comp_idx]
                time_series_name = actual[0].columns[comp_idx]

                # Plot actual sequence
                ax.plot(
                    actual[seq_idx].time_index,
                    actual[seq_idx].univariate_component(time_series_name).values(),
                    "b-",
                    label="Actual" if seq_idx == 0 else "_nolegend_",
                    linewidth=2,
                )

                # Plot predicted sequence
                ax.plot(
                    predictions[seq_idx].time_index,
                    predictions[seq_idx]
                    .univariate_component(time_series_name)
                    .values(),
                    "r-",
                    label="Forecast" if seq_idx == 0 else "_nolegend_",
                    linewidth=2,
                )

                # Customize plot
                if seq_idx == 0:
                    ax.set_title(f"{time_series_name}", pad=10)
                if seq_idx == n_rows - 1:
                    ax.set_xlabel("Time", fontsize=10)
                if comp_idx == 0:
                    ax.set_ylabel(f"Sequence {seq_idx + 1}", fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", which="major", labelsize=9)

        # Add legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            fancybox=True,
            shadow=True,
            fontsize=10,
        )

        # Adjust layout
        fig.suptitle(
            f"{model_name} Sequence Predictions (First {n_sequences} Sequences)",
            y=1.02,
            fontsize=12,
        )
        plt.tight_layout()

        # Log to wandb
        wandb.log({f"{model_name}_sequence_predictions": wandb.Image(fig)})

        # Close the figure
        plt.close(fig)

    def log_trial_metrics(
        self, metrics: dict[str, float], params: dict[str, Any], model_name: str
    ):
        """Log metrics and parameters for individual trials"""
        val_score = metrics.get("mae", 0)
        trial_num = metrics.get("trial_number", 0)

        # Log hyperparameter values for this trial
        for param_name, param_value in params.items():
            if param_name != "trial_number":
                wandb.log(
                    {
                        f"hyperparameters/{param_name}": param_value,
                        "trial_number": trial_num,
                    }
                )

        # Log validation score
        wandb.log({"optimization/val_score": val_score, "trial_number": trial_num})

        # Log scalar metrics
        wandb.log(
            {
                "metrics/min_val_error": min(
                    self.trial_history.get(model_name, {}).get("scores", [val_score])
                ),
                "metrics/training_time": metrics.get("training_time", 0),
                "metrics/rmse": metrics.get("rmse", 0),
                "metrics/mae": metrics.get("mae", 0),
            }
        )

    def _log_parameter_plots(self, model_name: str):
        """Create and log plots for each hyperparameter"""
        for param_name, history in self.param_history.items():
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(history["values"], history["scores"])
            plt.xlabel(param_name)
            plt.ylabel("MAE Score")
            plt.title(f"{param_name} vs MAE")

            if isinstance(history["values"][0], (int, float)):
                plt.grid(True)
            else:
                plt.xticks(rotation=45)

            wandb.log({f"{model_name}/parameter_plots/{param_name}": wandb.Image(fig)})
            plt.close(fig)

    def log_study_results(
        self, study: optuna.Study, model_name: str, dataset: str, horizon: int
    ):
        """Log final study results and visualizations"""
        try:
            # Create parallel coordinates plot using wandb
            param_names = list(study.best_params.keys())
            data = []
            for trial in study.trials:
                if trial.state == optuna.trial.TrialState.COMPLETE:
                    row = [trial.params[param] for param in param_names]
                    row.append(trial.value)  # Add val_score
                    data.append(row)

            wandb.log(
                {
                    "optimization/parallel_coordinates": wandb.plot.parallel_coordinates(
                        wandb.Table(data=data, columns=[*param_names, "val_score"]),
                        [*param_names, "val_score"],
                    )
                }
            )

            # Log Optuna visualizations
            history_fig = plot_optimization_history(study)
            importance_fig = plot_param_importances(study)
            parallel_fig = plot_parallel_coordinate(study)

            wandb.log(
                {
                    "optuna/optimization_history": history_fig,
                    "optuna/parameter_importance": importance_fig,
                    "optuna/parallel_coordinates": parallel_fig,
                }
            )
            plt.close("all")

        except Exception as e:
            wandb.log({f"{model_name}/visualization_error": str(e)})

    def log_optuna_study(
        self, study: optuna.Study, model_name: str, dataset: str, horizon: int
    ):
        """Create a callback for logging Optuna study progress"""

        def callback(study: optuna.Study, trial: optuna.Trial):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                # Log trial information with proper grouping
                wandb.log(
                    {
                        "optimization/current_trial_value": trial.value,
                        "optimization/best_value_so_far": study.best_value,
                        "trial_number": trial.number,
                    }
                )

        return callback
