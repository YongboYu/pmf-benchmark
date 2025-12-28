from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .utils.dfg_constructor import DFGConstructor
from .utils.er_calculator import calculate_entropic_relevance, calculate_er_metrics


@dataclass(frozen=True)
class EREvaluationConfig:
    dataset: str
    horizon_days: int
    start_time: str

    case_id_col: str = "case:concept:name"
    activity_col: str = "concept:name"
    timestamp_col: str = "time:timestamp"


def variant_log_from_sublog_df(
    sublog_df: pd.DataFrame,
    *,
    case_id_col: str = "case:concept:name",
    activity_col: str = "concept:name",
    timestamp_col: str = "time:timestamp",
) -> dict[tuple[str, ...], int]:
    """
    Build variant log (trace variants with multiplicities) from a sublog DataFrame.
    """
    if sublog_df.empty:
        return {}

    df = sublog_df.sort_values(by=[case_id_col, timestamp_col])
    grouped = df.groupby(case_id_col, sort=False)[activity_col].agg(list)

    variant_log: dict[tuple[str, ...], int] = {}
    for labels in grouped:
        key = tuple(str(x) for x in labels)
        if not key:
            continue
        variant_log[key] = variant_log.get(key, 0) + 1

    return variant_log


def evaluate_er_for_window(
    *,
    window_key: str,
    sublog: pd.DataFrame,
    dfgs: Mapping[str, Any],
    variant_log: Mapping[tuple[str, ...], int] | None = None,
    return_trace_counts: bool = False,
) -> dict[str, Any]:
    """
    Evaluate truth/pred/training ER for a single time window.
    """
    if variant_log is None:
        variant_log = variant_log_from_sublog_df(sublog)

    results: dict[str, Any] = {}
    for dfg_type in ("truth", "pred", "training"):
        dfg = dfgs.get(dfg_type, {"nodes": [], "arcs": []})
        if not dfg.get("nodes") or not dfg.get("arcs"):
            results[dfg_type] = {
                "entropic_relevance": float("nan"),
                "non_fitting_traces": 0,
                "total_traces": 0,
                "fitting_ratio": 0.0,
            }
            continue

        res = calculate_entropic_relevance(
            dfg,
            variant_log=variant_log,
            return_trace_counts=return_trace_counts,
        )
        results[dfg_type] = {
            "entropic_relevance": res["entropic_relevance"],
            "non_fitting_traces": res["non_fitting_traces"],
            "total_traces": res["total_traces"],
            "fitting_ratio": res["fitting_ratio"],
        }

    return results


def evaluate_rolling_er(
    *,
    combined_rolling_dfgs: Mapping[str, Mapping[str, Any]],
    seq_test_log: Mapping[str, pd.DataFrame] | None = None,
    variant_logs: Mapping[str, Mapping[tuple[str, ...], int]] | None = None,
    return_trace_counts: bool = False,
) -> dict[str, dict[str, Any]]:
    """
    Calculate ER for truth/pred/training DFGs across all rolling windows.
    """
    results: dict[str, dict[str, Any]] = {}

    for window_key in sorted(combined_rolling_dfgs.keys()):
        if variant_logs is not None:
            variant_log = variant_logs.get(window_key)
            if not variant_log:
                continue
            sublog = pd.DataFrame()  # unused when variant_log is provided
        else:
            if seq_test_log is None or window_key not in seq_test_log:
                continue
            sublog = seq_test_log[window_key]
            variant_log = variant_log_from_sublog_df(sublog)

        results[window_key] = evaluate_er_for_window(
            window_key=window_key,
            sublog=sublog,
            dfgs=combined_rolling_dfgs[window_key],
            variant_log=variant_log,
            return_trace_counts=return_trace_counts,
        )

    return results


def evaluate_er_end_to_end(
    *,
    cfg: EREvaluationConfig,
    log: Any,
    predictions_by_model: Mapping[str, pd.DataFrame],
    include_training: bool = True,
    return_trace_counts: bool = False,
) -> dict[str, Any]:
    """
    End-to-end ER evaluation for a dataset/horizon.

    `predictions_by_model` maps a model key (e.g., 'statistical_ar2') to a dataframe
    indexed by `sequence_start_time` (or convertible to that) with DF-relations columns.
    """
    constructor = DFGConstructor(
        case_id_col=cfg.case_id_col,
        activity_col=cfg.activity_col,
        timestamp_col=cfg.timestamp_col,
    )

    seq_test_log = constructor.extract_rolling_window_sublogs(
        log,
        start_time=cfg.start_time,
        time_length_days=cfg.horizon_days,
    )

    rolling_truth_dfgs = constructor.create_dfgs_from_rolling_window(seq_test_log)

    rolling_training_dfgs: dict[str, dict[str, Any]] = {}
    if include_training:
        rolling_training_dfgs = constructor.create_training_dfgs_for_windows(
            window_keys=seq_test_log.keys(),
            raw_log=log,
            training_ratio=0.8,
        )

    all_model_results: dict[str, Any] = {}
    all_window_metrics: dict[str, Any] = {}
    empty_dfg = {"nodes": [], "arcs": []}

    for model_key in sorted(predictions_by_model.keys()):
        pred_df = predictions_by_model[model_key]

        rolling_pred_dfgs: dict[str, dict[str, Any]] = {}
        for window_key in seq_test_log.keys():
            start_date = window_key.split("_")[0]
            if start_date not in pred_df.index:
                continue
            window_pred = pred_df.loc[[start_date]]
            rolling_pred_dfgs[window_key] = {
                "dfg_json": constructor.create_dfg_from_predictions(window_pred),
            }

        combined_rolling_dfgs: dict[str, dict[str, Any]] = {}
        all_windows = (
            set(rolling_truth_dfgs.keys())
            | set(rolling_pred_dfgs.keys())
            | set(rolling_training_dfgs.keys())
        )
        for window_key in all_windows:
            truth_json = rolling_truth_dfgs.get(window_key, {}).get(
                "dfg_json",
                empty_dfg,
            )
            pred_json = rolling_pred_dfgs.get(window_key, {}).get(
                "dfg_json",
                empty_dfg,
            )
            training_json = rolling_training_dfgs.get(window_key, {}).get(
                "dfg_json",
                empty_dfg,
            )
            combined_rolling_dfgs[window_key] = {
                "truth": truth_json,
                "pred": pred_json,
                "training": training_json,
            }

        rolling_er_results = evaluate_rolling_er(
            combined_rolling_dfgs=combined_rolling_dfgs,
            seq_test_log=seq_test_log,
            return_trace_counts=return_trace_counts,
        )
        metrics = calculate_er_metrics(rolling_er_results)
        all_model_results[model_key] = metrics
        all_window_metrics[model_key] = rolling_er_results

    return {
        "dataset": cfg.dataset,
        "horizon": cfg.horizon_days,
        "start_time": cfg.start_time,
        "models": all_model_results,
        "window_metrics": all_window_metrics,
    }
