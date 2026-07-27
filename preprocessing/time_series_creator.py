import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


class TimeSeriesCreator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def create_time_series(
        self,
        df_dict: dict,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        dataset: str,
    ):
        """Create and save time series data"""
        try:
            # Create time series
            df_series = self._create_df_time_series(df_dict, start_time, end_time)
            self._save_time_series(df_series, dataset)

            self.logger.info(f"Created time series for dataset: {dataset}")

        except Exception as e:
            self.logger.error(f"Error creating time series for {dataset}: {e}")
            raise

    def _create_df_time_series(
        self, df_dict: dict, start_time: pd.Timestamp, end_time: pd.Timestamp
    ) -> pd.DataFrame:
        """Create time series from DF relations"""
        # Create date range with daily frequency
        time_range = pd.date_range(
            start=start_time.normalize(),  # Start of day
            end=end_time.normalize(),  # End of day
            freq="D",
        )

        # Initialize DataFrame with zeros
        df_series = pd.DataFrame(0, index=time_range, columns=df_dict.keys())

        # Count occurrences for each DF relation per day
        for df_rel, occurrences in df_dict.items():
            for occ in occurrences:
                # Convert timestamp to datetime if string
                start_time = pd.to_datetime(occ["start_time"])
                # Get the day
                day = start_time.normalize()

                if day in df_series.index:
                    df_series.at[day, df_rel] += 1

        return df_series

    def _save_time_series(self, df_series: pd.DataFrame, dataset: str):
        """Save time series data"""
        base_path = Path(self.config["paths"]["processed"]["time_series"])
        base_path.mkdir(parents=True, exist_ok=True)

        # Path to H5 file - single file for all datasets
        h5_path = base_path / "time_series_df.h5"

        # Save to H5 file - create if not exists, append if exists
        if not h5_path.exists():
            # First time - create new file
            df_series.to_hdf(h5_path, key=dataset, mode="w", format="table")
        else:
            # File exists - append to it
            df_series.to_hdf(h5_path, key=dataset, mode="a", format="table")

        # Create dataset-specific directory
        dataset_path = base_path / dataset
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Save df_relations in dataset directory
        df_series.to_csv(dataset_path / "df_relations.csv")

        # Save metadata in dataset directory
        metadata = {
            "dataset": dataset,
            "start_date": df_series.index[0].strftime("%Y-%m-%d"),
            "end_date": df_series.index[-1].strftime("%Y-%m-%d"),
            "n_relations": len(df_series.columns),
            "frequency": "D",  # Daily frequency
        }

        with open(dataset_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Save stats in dataset directory
        stats = {
            "mean": df_series.mean().to_dict(),
            "std": df_series.std().to_dict(),
            "min": df_series.min().to_dict(),
            "max": df_series.max().to_dict(),
        }

        with open(dataset_path / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)

    def _save_multivariate_series(
        self, mv_series: dict[str, pd.DataFrame], dataset: str
    ):
        """Save multivariate time series"""
        base_path = Path(self.config["paths"]["processed"]["multivariate"]) / dataset
        base_path.mkdir(parents=True, exist_ok=True)

        # Save each feature set
        for feature_name, feature_df in mv_series.items():
            feature_df.to_csv(base_path / f"{feature_name}.csv")

        # Save metadata
        metadata = {
            "dataset": dataset,
            "features": list(mv_series.keys()),
            "feature_dimensions": {name: df.shape[1] for name, df in mv_series.items()},
            "frequency": self.config["time_series"]["interval"],
            "start_date": mv_series["df_patterns"].index[0].strftime("%Y-%m-%d"),
            "end_date": mv_series["df_patterns"].index[-1].strftime("%Y-%m-%d"),
        }

        with open(base_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
