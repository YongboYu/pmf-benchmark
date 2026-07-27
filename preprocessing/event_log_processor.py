import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pm4py


class EventLogProcessor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def process_log(self, dataset: str) -> pm4py.objects.log.obj.EventLog:
        """Process event log with filtering and trimming"""
        try:
            # Load original log
            input_path = Path(self.config["paths"]["raw"]) / f"{dataset}.xes"
            log = pm4py.read_xes(str(input_path))

            # Get original timespan
            org_timestamps = pm4py.get_event_attribute_values(log, "time:timestamp")
            org_start_time = min(org_timestamps)
            org_end_time = max(org_timestamps)

            # Filter infrequent variants
            filtered_log = self._filter_variants(log)

            # Add artificial start/end events
            processed_log = self._add_artificial_events(filtered_log)

            # Trim timespan
            trimmed_log = self._trim_timespan(processed_log)

            # Save processed log
            self._save_processed_log(trimmed_log, dataset)

            # Save processing information
            self._save_processing_info(
                dataset=dataset,
                org_log=log,
                filter_log=filtered_log,
                trimmed_log=trimmed_log,
                org_start_time=org_start_time,
                org_end_time=org_end_time,
                trim_start=self.get_log_timespan(trimmed_log)[0],
                trim_end=self.get_log_timespan(trimmed_log)[1],
            )

            return trimmed_log

        except Exception as e:
            self.logger.error(f"Error processing log {dataset}: {e}")
            raise

    def _filter_variants(self, log) -> pm4py.objects.log.obj.EventLog:
        """Filter infrequent variants"""
        return pm4py.filter_variants_by_coverage_percentage(
            log, min_coverage_percentage=self.config["event_log"]["filter_percentage"]
        )

    def _trim_timespan(self, log) -> pm4py.objects.log.obj.EventLog:
        """Trim log timespan"""
        start_time, end_time = self.get_log_timespan(log)
        trim_days = int(
            (end_time - start_time).days * self.config["event_log"]["trim_percentage"]
        )

        trim_start = start_time + pd.Timedelta(days=trim_days)
        trim_end = end_time - pd.Timedelta(days=trim_days)

        return pm4py.filter_time_range(
            log,
            trim_start.strftime("%Y-%m-%d %H:%M:%S"),
            trim_end.strftime("%Y-%m-%d %H:%M:%S"),
        )

    def _add_artificial_events(self, log) -> pm4py.objects.log.obj.EventLog:
        """Add artificial start/end events"""
        return pm4py.insert_artificial_start_end(log)

    def _save_processed_log(self, log, dataset: str):
        """Save processed log"""
        output_path = Path(self.config["paths"]["interim"]["processed_logs"])
        output_path.mkdir(parents=True, exist_ok=True)

        pm4py.write_xes(log, str(output_path / f"{dataset}.xes"))

    def _save_processing_info(
        self,
        dataset: str,
        org_log,
        filter_log,
        trimmed_log,
        org_start_time,
        org_end_time,
        trim_start,
        trim_end,
    ):
        """Save log processing information"""
        log_dir = Path("logs/data_preprocess")
        log_dir.mkdir(parents=True, exist_ok=True)

        # Get statistics for each log version
        org_stats = self._get_log_statistics(org_log)
        filter_stats = self._get_log_statistics(filter_log)
        trim_stats = self._get_log_statistics(trimmed_log)

        with open(log_dir / f"{dataset}.txt", "w") as f:
            f.write(f"Dataset: {dataset}\n\n")

            # Original log statistics
            f.write("Original Log Statistics:\n")
            f.write(f"Time length (days): {org_stats['time_length_days']}\n")
            f.write(f"Number of traces: {org_stats['num_traces']}\n")
            f.write(f"Number of events: {org_stats['num_events']}\n")
            f.write(f"Number of activities: {org_stats['num_activities']}\n")
            f.write(f"Number of variants: {org_stats['num_variants']}\n")
            f.write(f"Number of unique DFs: {org_stats['unique_dfs']}\n")
            f.write(f"Total DF occurrences: {org_stats['total_dfs']}\n")
            f.write(f"Start time: {org_stats['start_time']}\n")
            f.write(f"End time: {org_stats['end_time']}\n\n")

            # Filtered log statistics
            f.write("Filtered Log Statistics:\n")
            f.write(
                f"Filter percentage: {self.config['event_log']['filter_percentage']}\n"
            )
            f.write(
                f"Time length (days): {filter_stats['time_length_days']} "
                f"({round(filter_stats['time_length_days'] / org_stats['time_length_days'] * 100, 2)}%)\n"
            )
            f.write(
                f"Number of traces: {filter_stats['num_traces']} "
                f"({round(filter_stats['num_traces'] / org_stats['num_traces'] * 100, 2)}%)\n"
            )
            f.write(
                f"Number of events: {filter_stats['num_events']} "
                f"({round(filter_stats['num_events'] / org_stats['num_events'] * 100, 2)}%)\n"
            )
            f.write(
                f"Number of activities: {filter_stats['num_activities']} "
                f"({round(filter_stats['num_activities'] / org_stats['num_activities'] * 100, 2)}%)\n"
            )
            f.write(
                f"Number of variants: {filter_stats['num_variants']} "
                f"({round(filter_stats['num_variants'] / org_stats['num_variants'] * 100, 2)}%)\n"
            )
            f.write(
                f"Number of unique DFs: {filter_stats['unique_dfs']} "
                f"({round(filter_stats['unique_dfs'] / org_stats['unique_dfs'] * 100, 2)}%)\n"
            )
            f.write(
                f"Total DF occurrences: {filter_stats['total_dfs']} "
                f"({round(filter_stats['total_dfs'] / org_stats['total_dfs'] * 100, 2)}%)\n\n"
            )

            # Trimmed log statistics
            f.write("Trimmed Log Statistics:\n")
            f.write(f"Trim percentage: {self.config['event_log']['trim_percentage']}\n")
            f.write(
                f"Time length (days): {trim_stats['time_length_days']} "
                f"({round(trim_stats['time_length_days'] / org_stats['time_length_days'] * 100, 2)}%)\n"
            )
            f.write(
                f"Number of traces: {trim_stats['num_traces']} "
                f"({round(trim_stats['num_traces'] / org_stats['num_traces'] * 100, 2)}%)\n"
            )
            f.write(
                f"Number of events: {trim_stats['num_events']} "
                f"({round(trim_stats['num_events'] / org_stats['num_events'] * 100, 2)}%)\n"
            )
            f.write(
                f"Number of activities: {trim_stats['num_activities']} "
                f"({round(trim_stats['num_activities'] / org_stats['num_activities'] * 100, 2)}%)\n"
            )
            f.write(
                f"Number of variants: {trim_stats['num_variants']} "
                f"({round(trim_stats['num_variants'] / org_stats['num_variants'] * 100, 2)}%)\n"
            )
            f.write(
                f"Number of unique DFs: {trim_stats['unique_dfs']} "
                f"({round(trim_stats['unique_dfs'] / org_stats['unique_dfs'] * 100, 2)}%)\n"
            )
            f.write(
                f"Total DF occurrences: {trim_stats['total_dfs']} "
                f"({round(trim_stats['total_dfs'] / org_stats['total_dfs'] * 100, 2)}%)\n"
            )
            f.write(f"Start time: {trim_stats['start_time']}\n")
            f.write(f"End time: {trim_stats['end_time']}\n")

    def get_log_timespan(self, log) -> tuple[datetime, datetime]:
        """Get log start and end times"""
        timestamps = pm4py.get_event_attribute_values(log, "time:timestamp")
        return min(timestamps), max(timestamps)

    def _get_log_statistics(self, log) -> dict[str, Any]:
        """Get comprehensive statistics about an event log"""
        try:
            # Get timespan
            timestamps = pm4py.get_event_attribute_values(log, "time:timestamp")
            start_time = min(timestamps)
            end_time = max(timestamps)
            time_length = (end_time - start_time).days

            # Get basic counts
            num_traces = len(log)
            num_events = sum(len(trace) for trace in log)

            # Get unique activities
            activities = pm4py.get_event_attribute_values(log, "concept:name")
            num_activities = len(activities)

            # Get variants
            variants = pm4py.get_variants(log)
            num_variants = len(variants)

            # Get directly-follows relations
            dfg, _start_activities, _end_activities = pm4py.discover_dfg(
                log,
                case_id_key="case:concept:name",
                activity_key="concept:name",
                timestamp_key="time:timestamp",
            )
            unique_dfs = len(dfg)
            total_dfs = sum(dfg.values())

            # # Create DataFrame and sort by frequency in descending order
            # df_table = pd.DataFrame(dfg_data)
            # df_table = df_table.sort_values(by='frequency', ascending=False)
            #
            # # Save to CSV
            # output_path = Path(self.config['paths']['interim']['dfg_tables'])
            # output_path.mkdir(parents=True, exist_ok=True)
            # df_table.to_csv(output_path / f'{dataset}_dfg.csv', index=False)
            #
            # # Also save as Excel for better visualization
            # df_table.to_excel(output_path / f'{dataset}_dfg.xlsx', index=False)

            return {
                "time_length_days": time_length,
                "num_traces": num_traces,
                "num_events": num_events,
                "num_activities": num_activities,
                "num_variants": num_variants,
                "unique_dfs": unique_dfs,
                "total_dfs": total_dfs,
                "start_time": start_time,
                "end_time": end_time,
            }
        except Exception as e:
            self.logger.error(f"Error getting log statistics: {e}")
            raise
