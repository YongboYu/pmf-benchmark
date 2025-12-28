from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class DFGConstructor:
    """
    Fast(er) DFG/log utilities for ER evaluation.

    Notes:
    - This module is written to work with PM4Py when installed, but keeps the
      core logic (like window extraction and variant aggregation) in pandas for speed.
    """

    case_id_col: str = "case:concept:name"
    activity_col: str = "concept:name"
    timestamp_col: str = "time:timestamp"

    @staticmethod
    def _ensure_datetime(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        return df

    def to_dataframe(self, log: Any) -> pd.DataFrame:
        if isinstance(log, pd.DataFrame):
            return self._ensure_datetime(log, self.timestamp_col)
        try:
            import pm4py  # type: ignore

            df = pm4py.convert_to_dataframe(log)
            return self._ensure_datetime(df, self.timestamp_col)
        except Exception as e:  # pragma: no cover
            raise TypeError(
                "Unsupported log type; expected pandas DataFrame or PM4Py EventLog",
            ) from e

    def extract_time_period_sublog(
        self,
        log: Any,
        *,
        start_time: pd.Timestamp | str,
        end_time: pd.Timestamp | str,
    ) -> pd.DataFrame:
        """
        Vectorized sublog extraction:
        Keep events in cases where:
        - event within [start, end], or
        - next event within [start, end]
        """
        df = self.to_dataframe(log)
        df = df.sort_values(by=[self.case_id_col, self.timestamp_col])

        start = pd.to_datetime(start_time)
        end = pd.to_datetime(end_time)

        # Align timezone if needed (best effort).
        sample_ts = df[self.timestamp_col].iloc[0]
        if (
            getattr(sample_ts, "tz", None) is not None
            and getattr(
                start,
                "tzinfo",
                None,
            )
            is None
        ):
            start = start.tz_localize(sample_ts.tz)
            end = end.tz_localize(sample_ts.tz)

        is_within = (df[self.timestamp_col] >= start) & (df[self.timestamp_col] <= end)
        next_ts = df.groupby(self.case_id_col, sort=False)[self.timestamp_col].shift(-1)
        next_within = (next_ts >= start) & (next_ts <= end)

        mask = is_within | next_within
        result = df.loc[mask, df.columns].copy()

        # Replace special symbols with text representations (legacy compatibility).
        result[self.activity_col] = result[self.activity_col].replace(
            {"▶": "Start", "■": "End"},
        )
        return result

    def extract_training_log(
        self,
        log: Any,
        *,
        training_ratio: float = 0.8,
    ) -> tuple[Any, int]:
        """
        Extract the training log as the first `training_ratio` of the full period
        (by day).

        Returns the training log in the same "kind" as input when possible
        (PM4Py EventLog vs DataFrame), and the training period length in days.
        """
        df = self.to_dataframe(log)
        min_date = df[self.timestamp_col].min().date()
        max_date = df[self.timestamp_col].max().date()
        total_days = (max_date - min_date).days + 1
        days_for_training = int(total_days * training_ratio)
        cutoff_date = min_date + pd.Timedelta(days=days_for_training)

        training_df = df[df[self.timestamp_col].dt.date < cutoff_date].copy()
        training_df[self.activity_col] = training_df[self.activity_col].replace(
            {"▶": "Start", "■": "End"},
        )

        if isinstance(log, pd.DataFrame):
            return training_df, days_for_training

        try:
            import pm4py  # type: ignore

            parameters = {
                self.case_id_col: self.case_id_col,
                self.activity_col: self.activity_col,
                self.timestamp_col: self.timestamp_col,
            }
            return (
                pm4py.convert_to_event_log(training_df, parameters=parameters),
                days_for_training,
            )
        except Exception:
            # Fallback: return DataFrame if PM4Py conversion isn't available.
            return training_df, days_for_training

    def extract_rolling_window_sublogs(
        self,
        log: Any,
        *,
        start_time: pd.Timestamp | str,
        time_length_days: int,
    ) -> dict[str, pd.DataFrame]:
        df = self.to_dataframe(log)
        df = df.sort_values(by=[self.case_id_col, self.timestamp_col])

        time_length_days = int(time_length_days)
        start_time = pd.to_datetime(start_time)

        sample_ts = df[self.timestamp_col].iloc[0]
        if (
            getattr(sample_ts, "tz", None) is not None
            and getattr(
                start_time,
                "tzinfo",
                None,
            )
            is None
        ):
            start_time = start_time.tz_localize(sample_ts.tz)

        data_end_time = df[self.timestamp_col].max()

        time_delta = data_end_time - start_time
        num_windows = max(1, time_delta.days - time_length_days + 2)

        sublogs: dict[str, pd.DataFrame] = {}
        for i in range(num_windows):
            window_start = start_time + pd.Timedelta(days=i)
            window_end = window_start + pd.Timedelta(days=time_length_days - 1)

            window_key = (
                f"{window_start.strftime('%Y-%m-%d')}_{window_end.strftime('%Y-%m-%d')}"
            )
            sublog = self.extract_time_period_sublog(
                df,
                start_time=window_start,
                end_time=window_end,
            )
            if len(sublog) == 0:
                continue

            # Add artificial start/end markers if PM4Py is available.
            try:
                import pm4py  # type: ignore

                sublog = pm4py.insert_artificial_start_end(sublog)
            except Exception:
                pass

            sublogs[window_key] = sublog

        return sublogs

    @staticmethod
    def dfg_to_json(
        dfg_truth: Mapping[tuple[str, str], int],
    ) -> dict[str, Any]:
        reverse_map: dict[str, int] = {"▶": 0, "■": 1}

        for (source, target), _freq in dfg_truth.items():
            if source not in reverse_map:
                reverse_map[source] = len(reverse_map)
            if target not in reverse_map:
                reverse_map[target] = len(reverse_map)

        arcs = []
        node_freq: dict[str, int] = {node: 0 for node in reverse_map.keys()}

        for (source, target), freq in dfg_truth.items():
            arcs.append(
                {
                    "from": reverse_map[source],
                    "to": reverse_map[target],
                    "freq": int(freq),
                },
            )
            if source == "▶":
                node_freq[source] += int(freq)
            else:
                node_freq[target] += int(freq)

        nodes = [
            {"label": node, "id": node_id, "freq": int(node_freq.get(node, 0))}
            for node, node_id in reverse_map.items()
        ]
        return {"nodes": nodes, "arcs": arcs}

    def create_dfgs_from_rolling_window(
        self,
        seq_test_log: dict[str, pd.DataFrame],
    ) -> dict[str, dict[str, Any]]:
        try:
            import pm4py  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PM4Py is required to discover DFGs from logs") from e

        dfgs_dict: dict[str, dict[str, Any]] = {}
        for window_key, sublog in seq_test_log.items():
            dfg, _, _ = pm4py.discover_dfg(sublog)
            dfgs_dict[window_key] = {
                "dfg": dfg,
                "dfg_json": self.dfg_to_json(dfg),
                "sublog": sublog,
            }
        return dfgs_dict

    def create_training_dfgs_for_windows(
        self,
        *,
        window_keys: Iterable[str],
        raw_log: Any,
        training_ratio: float = 0.8,
    ) -> dict[str, dict[str, Any]]:
        """
        Create (and reuse) a training baseline DFG for all windows.
        """
        try:
            import pm4py  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("PM4Py is required to discover DFGs from logs") from e

        training_log, _days_for_training = self.extract_training_log(
            raw_log,
            training_ratio=training_ratio,
        )
        training_log = pm4py.insert_artificial_start_end(training_log)
        dfg, _, _ = pm4py.discover_dfg(training_log)
        dfg_json = self.dfg_to_json(dfg)

        out: dict[str, dict[str, Any]] = {}
        for window_key in window_keys:
            out[window_key] = {
                "dfg": dfg,
                "dfg_json": dfg_json,
                "training_log": training_log,
            }
        return out

    @staticmethod
    def create_dfg_from_predictions(window_pred: pd.DataFrame) -> dict[str, Any]:
        reverse_map: dict[str, int] = {"▶": 0, "■": 1, "Start": 2, "End": 3}
        original_activities: set[str] = set()

        # Identify activities with non-zero predicted frequency.
        for relation in window_pred.columns:
            if "->" not in relation:
                continue
            src, dst = [part.strip() for part in relation.split("->")]
            src = src.replace("▶", "Start").replace("■", "End")
            dst = dst.replace("▶", "Start").replace("■", "End")
            total = float(window_pred[relation].sum())
            if total <= 0:
                continue
            original_activities.add(src)
            original_activities.add(dst)
            if src not in reverse_map:
                reverse_map[src] = len(reverse_map)
            if dst not in reverse_map:
                reverse_map[dst] = len(reverse_map)

        arcs: list[dict[str, Any]] = []
        node_freq: dict[str, float] = {node: 0.0 for node in reverse_map.keys()}

        for relation in window_pred.columns:
            if "->" not in relation:
                continue
            src, dst = [part.strip() for part in relation.split("->")]
            src = src.replace("▶", "Start").replace("■", "End")
            dst = dst.replace("▶", "Start").replace("■", "End")
            if src not in reverse_map or dst not in reverse_map:
                continue
            total_freq = int(round(float(window_pred[relation].sum())))
            if total_freq <= 0:
                continue
            arcs.append(
                {
                    "from": reverse_map[src],
                    "to": reverse_map[dst],
                    "freq": total_freq,
                },
            )
            node_freq[dst] = node_freq.get(dst, 0.0) + total_freq

        # Connect '▶' and '■' to each activity (legacy behavior).
        for activity in sorted(original_activities):
            arcs.append(
                {"from": reverse_map["▶"], "to": reverse_map[activity], "freq": 1},
            )
            node_freq["▶"] = node_freq.get("▶", 0.0) + 1

        for activity in sorted(original_activities):
            arcs.append(
                {"from": reverse_map[activity], "to": reverse_map["■"], "freq": 1},
            )
            node_freq["■"] = node_freq.get("■", 0.0) + 1

        nodes = [
            {
                "label": node,
                "id": node_id,
                "freq": int(round(node_freq.get(node, 0.0))),
            }
            for node, node_id in reverse_map.items()
        ]
        return {"nodes": nodes, "arcs": arcs}
