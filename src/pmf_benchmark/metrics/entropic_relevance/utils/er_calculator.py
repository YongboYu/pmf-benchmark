from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .background_model import BackgroundModel, TraceKey


@dataclass(frozen=True)
class Automaton:
    sources: tuple[int, ...]
    trans_table: Mapping[tuple[int, str], tuple[int, float]]


def _build_node_info(nodes: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    return {int(node["id"]): str(node["label"]) for node in nodes}


def dfg_to_automaton_without_end_counts(dfg: Mapping[str, Any]) -> Automaton:
    """
    Convert a DFG (nodes/arcs JSON) to an automaton, ignoring '■' transitions when
    normalizing outgoing probabilities.

    This matches the intent of
    `ER_v2.utils_ER_v2.ERCalculator.convert_dfg_into_automaton_without_end_counts`
    but is implemented in a smaller, deterministic, and dependency-free form.
    """
    nodes = dfg.get("nodes", []) or []
    arcs = dfg.get("arcs", []) or []
    node_info = _build_node_info(nodes)

    sinks = set(node_info.keys())
    sources = set(node_info.keys())

    agg_outgoing_frequency: dict[int, int] = {}

    for arc in arcs:
        freq = int(arc.get("freq", 0) or 0)
        if freq <= 0:
            continue

        from_node = int(arc["from"])
        to_node = int(arc["to"])

        if node_info.get(to_node) != "■":
            agg_outgoing_frequency[from_node] = (
                agg_outgoing_frequency.get(from_node, 0) + freq
            )

        sinks.discard(from_node)
        sources.discard(to_node)

    transitions: dict[tuple[int, str], tuple[int, float]] = {}

    for arc in arcs:
        freq = int(arc.get("freq", 0) or 0)
        if freq <= 0:
            continue

        from_node = int(arc["from"])
        to_node = int(arc["to"])
        label = node_info.get(to_node)
        if label is None:
            continue

        from_label = node_info.get(from_node)

        if from_label == "▶" or label == "■":
            transitions[(from_node, label)] = (to_node, 0.0)
            continue

        if to_node in sinks:
            continue

        denom = agg_outgoing_frequency.get(from_node, 0)
        if denom <= 0:
            continue

        prob = freq / denom
        if prob <= 0.0:
            continue
        transitions[(from_node, label)] = (to_node, math.log2(prob))

    pruned_sources = tuple(sorted(sources))
    return Automaton(sources=pruned_sources, trans_table=transitions)


def variants_from_traces(
    traces: Iterable[Sequence[Mapping[str, Any]]],
) -> dict[TraceKey, int]:
    """
    Build a variant log from a list of traces.

    Each trace is a list of events with a `concept:name` key.
    """
    variant_log: dict[TraceKey, int] = {}
    for trace in traces:
        labels = tuple(
            str(evt["concept:name"])
            for evt in trace
            if isinstance(evt, Mapping) and "concept:name" in evt
        )
        if not labels:
            continue
        variant_log[labels] = variant_log.get(labels, 0) + 1
    return variant_log


def calculate_entropic_relevance(
    dfg: Mapping[str, Any],
    *,
    variant_log: Mapping[TraceKey, int],
    return_trace_counts: bool = False,
) -> dict[str, Any]:
    """
    Compute entropic relevance for a DFG against a (possibly aggregated) variant log.

    Returns a dictionary mirroring the parts of ER_v2 outputs that are used downstream,
    without emitting per-trace prints.
    """
    automaton = dfg_to_automaton_without_end_counts(dfg)
    if not automaton.sources:
        return {
            "entropic_relevance": float("nan"),
            "non_fitting_traces": 0,
            "total_traces": 0,
            "fitting_ratio": 0.0,
            "fitting_traces": {} if return_trace_counts else None,
            "non_fitting_traces_by_variant": {} if return_trace_counts else None,
        }

    # Precompute per-variant labels for deterministic performance.
    variant_labels: dict[TraceKey, tuple[str, ...]] = {
        k: tuple(lbl for lbl in k if lbl != "▶") for k in variant_log
    }

    best: dict[str, Any] | None = None
    for source in automaton.sources:
        model = BackgroundModel()
        fitting_traces: dict[str, int] = {}
        non_fitting_traces: dict[str, int] = {}

        for trace_key, freq in variant_log.items():
            labels = variant_labels[trace_key]
            if not labels:
                continue

            curr = source
            log2_prob = 0.0
            non_fitting = False

            # Equivalent to the legacy length counting: counts all labels except '▶'.
            trace_length = len(labels)

            for label in labels:
                if not non_fitting and (curr, label) in automaton.trans_table:
                    curr, step_log2 = automaton.trans_table[(curr, label)]
                    log2_prob += step_log2
                else:
                    non_fitting = True

            is_fitting = (not non_fitting) and (labels[-1] == "■")

            if is_fitting:
                model.observe_variant(
                    trace_key,
                    freq,
                    trace_length=trace_length,
                    log2_probability=log2_prob,
                    labels=labels,
                )
                if return_trace_counts:
                    key = "_".join(labels)
                    fitting_traces[key] = fitting_traces.get(key, 0) + freq
            else:
                model.observe_variant(
                    trace_key,
                    freq,
                    trace_length=trace_length,
                    log2_probability=None,
                    labels=labels,
                )
                if return_trace_counts:
                    key = "_".join(labels)
                    non_fitting_traces[key] = non_fitting_traces.get(key, 0) + freq

        er = model.compute_relevance()
        total = model.number_of_traces
        non_fit = model.total_number_non_fitting_traces
        fitting_ratio = 1.0 - (non_fit / total) if total > 0 else 0.0

        candidate = {
            "entropic_relevance": er,
            "non_fitting_traces": non_fit,
            "total_traces": total,
            "fitting_ratio": fitting_ratio,
            "fitting_traces": fitting_traces if return_trace_counts else None,
            "non_fitting_traces_by_variant": (
                non_fitting_traces if return_trace_counts else None
            ),
        }

        if best is None:
            best = candidate
        elif not math.isnan(er) and (
            math.isnan(best["entropic_relevance"]) or er < best["entropic_relevance"]
        ):
            best = candidate

    return best or {
        "entropic_relevance": float("nan"),
        "non_fitting_traces": 0,
        "total_traces": 0,
        "fitting_ratio": 0.0,
        "fitting_traces": {} if return_trace_counts else None,
        "non_fitting_traces_by_variant": {} if return_trace_counts else None,
    }


def calculate_er_metrics(
    rolling_er_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """
    Compute MAE/RMSE/MAPE between truth and prediction ER values across comparable
    windows.
    """
    truth_values: list[float] = []
    pred_values: list[float] = []

    for window_key in sorted(rolling_er_results.keys()):
        results = rolling_er_results[window_key]
        truth_er = float(results["truth"]["entropic_relevance"])
        pred_er = float(results["pred"]["entropic_relevance"])
        if math.isnan(truth_er) or math.isnan(pred_er):
            continue
        truth_values.append(truth_er)
        pred_values.append(pred_er)

    if not truth_values:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}

    n = len(truth_values)
    mae = sum(abs(t - p) for t, p in zip(truth_values, pred_values, strict=True)) / n
    rmse = math.sqrt(
        sum((t - p) ** 2 for t, p in zip(truth_values, pred_values, strict=True)) / n
    )

    mape_values = [
        abs((t - p) / t)
        for t, p in zip(truth_values, pred_values, strict=True)
        if t != 0.0
    ]
    mape = (sum(mape_values) / len(mape_values) * 100) if mape_values else float("nan")

    return {"n": n, "mae": mae, "rmse": rmse, "mape": mape}
