from __future__ import annotations

import json
import os
import pickle
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .er_evaluation import variant_log_from_sublog_df
from .utils.dfg_constructor import DFGConstructor
from .utils.er_calculator import calculate_entropic_relevance, calculate_er_metrics

DEFAULT_START_TIMES: dict[str, str] = {
    "BPI2017": "2016-10-22 00:00:00",
    "BPI2019_1": "2018-10-11 00:00:00",
    "Hospital_Billing": "2015-02-05 00:00:00",
    "sepsis": "2015-01-05 00:00:00",
    "Sepsis": "2015-01-05 00:00:00",
}


@dataclass(frozen=True)
class ERFastConfig:
    dataset: str
    horizon: int
    start_time: str | None = None

    log_file: str | None = None
    predictions_root: str | None = None
    output_dir: str = "results/er_metrics_v2_fast"
    cache_dir: str | None = None
    no_cache: bool = False
    models: list[str] | None = None


def _safe_version(pkg: str) -> str | None:
    try:
        import importlib.metadata as im

        return im.version(pkg)
    except Exception:
        return None


def _load_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(pred_path)

    if isinstance(df.index, pd.MultiIndex) and "sequence_start_time" in (
        df.index.names or []
    ):
        agg = df.groupby(level="sequence_start_time").sum(numeric_only=True)
    elif "sequence_start_time" in df.columns:
        agg = df.groupby("sequence_start_time").sum(numeric_only=True)
    else:
        agg = df.copy()

    try:
        idx = pd.to_datetime(agg.index)
        agg.index = idx.strftime("%Y-%m-%d")
    except Exception:
        agg.index = agg.index.astype(str)

    agg = agg.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    agg = agg.round(0).clip(lower=0)
    return agg.astype("int64")


def _discover_prediction_files(predictions_root: Path) -> dict[str, Path]:
    model_files: dict[str, Path] = {}
    for path in sorted(predictions_root.glob("*/*_all_predictions.parquet")):
        model_group = path.parent.name
        model_name = path.name.replace("_all_predictions.parquet", "")
        model_key = f"{model_group}_{model_name}"
        model_files[model_key] = path
    return model_files


def _cache_key(log_file: Path, *, dataset: str, horizon: int, start_time: str) -> str:
    stat = log_file.stat()
    return (
        f"{dataset}_h{horizon}_start_{start_time}_mtime_{int(stat.st_mtime)}"
        f"_size_{stat.st_size}"
    )


def _load_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _save_cache(cache_path: Path, obj: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=4)
    tmp.replace(cache_path)


def _resolve_log_file(dataset: str, log_file: str | None) -> Path:
    candidates = [
        Path(log_file) if log_file else None,
        Path(f"data/interim/processed_logs/{dataset}.xes"),
        Path(f"data/interim/processed_logs/{dataset}.xes.gz"),
        Path(f"data/external/{dataset}.xes"),
        Path(f"data/external/{dataset}.xes.gz"),
    ]
    chosen = next((p for p in candidates if p is not None and p.exists()), None)
    if chosen is None:
        raise FileNotFoundError(
            "Log file not found. Tried: "
            f"{[str(p) for p in candidates if p is not None]}",
        )
    return chosen


def run_er_evaluation_fast(cfg: ERFastConfig) -> dict[str, Any]:
    """
    Run fast ER evaluation end-to-end.

    Returns a dict with output file paths and the cache path.
    """
    dataset = cfg.dataset
    horizon = int(cfg.horizon)
    start_time = cfg.start_time or DEFAULT_START_TIMES.get(dataset)
    if not start_time:
        raise ValueError(
            f"No default start time for dataset '{dataset}'. Provide start_time.",
        )

    log_file = _resolve_log_file(dataset, cfg.log_file)

    predictions_root = Path(
        cfg.predictions_root or f"results/{dataset}/horizon_{horizon}/predictions",
    )
    if not predictions_root.exists():
        raise FileNotFoundError(f"Predictions root not found: {predictions_root}")

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(cfg.cache_dir or f"results/{dataset}/horizon_{horizon}/er_cache")
    cache_key = _cache_key(
        log_file,
        dataset=dataset,
        horizon=horizon,
        start_time=start_time,
    )
    cache_path = cache_dir / f"{cache_key}.pkl"

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str((Path.cwd() / ".cache" / "matplotlib").resolve()),
    )

    try:
        import pm4py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pm4py is required for ER evaluation from XES logs. "
            "Install dependencies and retry.",
        ) from e

    model_files = _discover_prediction_files(predictions_root)
    if cfg.models:
        missing = [m for m in cfg.models if m not in model_files]
        if missing:
            raise FileNotFoundError(
                f"Requested models not found under {predictions_root}: {missing}",
            )
        model_files = {k: model_files[k] for k in cfg.models}

    if not model_files:
        raise FileNotFoundError(
            f"No prediction files found under {predictions_root} "
            "(expected */*_all_predictions.parquet)",
        )

    cached = None if cfg.no_cache else _load_cache(cache_path)

    constructor = DFGConstructor()
    if cached is None:
        log = pm4py.read_xes(str(log_file))
        seq_test_log = constructor.extract_rolling_window_sublogs(
            log,
            start_time=start_time,
            time_length_days=horizon,
        )

        rolling_truth_dfgs = constructor.create_dfgs_from_rolling_window(seq_test_log)
        rolling_training_dfgs = constructor.create_training_dfgs_for_windows(
            window_keys=seq_test_log.keys(),
            raw_log=log,
            training_ratio=0.8,
        )

        variant_logs: dict[str, dict[tuple[str, ...], int]] = {
            window_key: variant_log_from_sublog_df(sublog)
            for window_key, sublog in seq_test_log.items()
        }

        cached = {
            "dataset": dataset,
            "horizon": horizon,
            "start_time": start_time,
            "window_keys": sorted(seq_test_log.keys()),
            "variant_logs": variant_logs,
            "truth_dfg_json": {k: v["dfg_json"] for k, v in rolling_truth_dfgs.items()},
            "training_dfg_json": {
                k: v["dfg_json"] for k, v in rolling_training_dfgs.items()
            },
        }
        if not cfg.no_cache:
            _save_cache(cache_path, cached)

    window_keys = list(cached["window_keys"])
    variant_logs = cached["variant_logs"]
    empty_dfg = {"nodes": [], "arcs": []}

    if "truth_metrics" not in cached or "training_metrics" not in cached:
        truth_metrics: dict[str, dict[str, Any]] = {}
        training_metrics: dict[str, dict[str, Any]] = {}

        for window_key in window_keys:
            vl = variant_logs.get(window_key)
            if not vl:
                continue
            truth_dfg = cached["truth_dfg_json"].get(window_key, empty_dfg)
            training_dfg = cached["training_dfg_json"].get(window_key, empty_dfg)

            truth_metrics[window_key] = calculate_entropic_relevance(
                truth_dfg,
                variant_log=vl,
            )
            training_metrics[window_key] = calculate_entropic_relevance(
                training_dfg,
                variant_log=vl,
            )

        cached["truth_metrics"] = truth_metrics
        cached["training_metrics"] = training_metrics
        if not cfg.no_cache:
            _save_cache(cache_path, cached)

    truth_metrics = cached["truth_metrics"]
    training_metrics = cached["training_metrics"]

    model_metrics: dict[str, dict[str, float]] = {}
    per_window_table: dict[str, dict[str, Any]] = {}

    for model_key, pred_path in sorted(model_files.items()):
        pred_df = _load_predictions(pred_path)

        rolling_results: dict[str, dict[str, Any]] = {}
        for window_key in window_keys:
            vl = variant_logs.get(window_key)
            if not vl:
                continue

            start_date = window_key.split("_")[0]
            pred_json = empty_dfg
            if start_date in pred_df.index:
                pred_json = constructor.create_dfg_from_predictions(
                    pred_df.loc[[start_date]],
                )

            pred_res = calculate_entropic_relevance(pred_json, variant_log=vl)

            rolling_results[window_key] = {
                "truth": truth_metrics[window_key],
                "pred": pred_res,
                "training": training_metrics[window_key],
            }

        metrics = calculate_er_metrics(rolling_results)
        model_metrics[model_key] = metrics

        for window_key, res in rolling_results.items():
            row = per_window_table.setdefault(
                window_key,
                {
                    "truth_er": res["truth"]["entropic_relevance"],
                    "truth_fitting_ratio": res["truth"]["fitting_ratio"],
                    "truth_total_traces": res["truth"]["total_traces"],
                    "training_er": res["training"]["entropic_relevance"],
                    "training_fitting_ratio": res["training"]["fitting_ratio"],
                    "training_total_traces": res["training"]["total_traces"],
                },
            )
            row[f"{model_key}_er"] = res["pred"]["entropic_relevance"]
            row[f"{model_key}_fitting_ratio"] = res["pred"]["fitting_ratio"]
            row[f"{model_key}_total_traces"] = res["pred"]["total_traces"]

    summary_rows = [
        {"model": model_key, **m} for model_key, m in sorted(model_metrics.items())
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / f"{dataset}_horizon_{horizon}_er_metrics_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    window_df = pd.DataFrame.from_dict(per_window_table, orient="index")
    window_df.index.name = "window"
    window_csv = output_dir / f"{dataset}_horizon_{horizon}_combined_er_metrics.csv"
    window_df.to_csv(window_csv)

    combined_json = output_dir / f"{dataset}_horizon_{horizon}_combined_er_metrics.json"
    with combined_json.open("w") as f:
        json.dump(
            {
                "dataset": dataset,
                "horizon": horizon,
                "start_time": start_time,
                "models": model_metrics,
                "window_metrics": per_window_table,
                "metadata": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "pandas": _safe_version("pandas"),
                    "pm4py": _safe_version("pm4py"),
                    "pyarrow": _safe_version("pyarrow"),
                },
            },
            f,
            indent=2,
        )

    return {
        "summary_csv": str(summary_csv),
        "window_csv": str(window_csv),
        "combined_json": str(combined_json),
        "cache_path": str(cache_path),
    }
