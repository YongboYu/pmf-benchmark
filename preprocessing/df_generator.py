import json
import logging
from pathlib import Path
from typing import Any

import pm4py
from pm4py.objects.log.obj import EventLog


class DFGenerator:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def generate_df_relations(self, log, dataset: str):
        """Generate and save DF relations"""
        try:
            # Ensure log is an EventLog object
            if not isinstance(log, EventLog):
                log = pm4py.convert_to_event_log(log)

            # Extract DF relations
            df_dict = self._extract_df_relations(log)

            # Save DF relations
            self._save_df_relations(df_dict, dataset)

            return df_dict

        except Exception as e:
            self.logger.error(f"Error generating DF relations for {dataset}: {e}")
            raise

    def _extract_df_relations(self, log) -> dict[str, list[dict]]:
        """Extract directly-follows relations from log"""
        df_dict = {}

        # Iterate over each trace (case) in the log
        for trace in log:
            # Iterate over pairs of consecutive events in the trace
            for i in range(len(trace) - 1):
                event_a = trace[i]
                event_b = trace[i + 1]

                # Extract activity names and timestamps
                activity_a = event_a[self.config["event_log"]["columns"]["activity"]]
                activity_b = event_b[self.config["event_log"]["columns"]["activity"]]
                start_time = event_a[self.config["event_log"]["columns"]["timestamp"]]
                end_time = event_b[self.config["event_log"]["columns"]["timestamp"]]

                # Calculate duration
                duration = end_time - start_time

                # Create DF relation key with space around arrow
                df_key = f"{activity_a} -> {activity_b}"

                # Initialize list if key doesn't exist
                if df_key not in df_dict:
                    df_dict[df_key] = []

                # Append relation information
                df_dict[df_key].append(
                    {
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                    }
                )

        return df_dict

    def _save_df_relations(self, df_dict: dict[str, list[dict]], dataset: str):
        """Save DF relations to JSON"""
        output_path = Path(self.config["paths"]["interim"]["df_relations"])
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / f"{dataset}.json", "w") as f:
            json.dump(df_dict, f, default=str, indent=4)
