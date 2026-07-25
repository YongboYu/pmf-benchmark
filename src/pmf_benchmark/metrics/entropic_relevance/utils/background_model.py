from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field

TraceKey = tuple[str, ...]


@dataclass
class BackgroundModel:
    """
    Aggregates per-trace/variant statistics and computes entropic relevance.

    This is a refactored, deterministic, variant-aggregated implementation of the
    legacy ER_v2 `BackgroundModel`.
    """

    trace_frequency: dict[TraceKey, int] = field(default_factory=dict)
    trace_size: dict[TraceKey, int] = field(default_factory=dict)
    log2_of_model_probability: dict[TraceKey, float] = field(default_factory=dict)
    labels: set[str] = field(default_factory=set)
    number_of_traces: int = 0
    total_number_non_fitting_traces: int = 0

    def observe_variant(
        self,
        trace_key: TraceKey,
        frequency: int,
        *,
        trace_length: int,
        log2_probability: float | None,
        labels: Iterable[str],
    ) -> None:
        if frequency <= 0:
            return

        self.number_of_traces += frequency
        self.trace_frequency[trace_key] = (
            self.trace_frequency.get(trace_key, 0) + frequency
        )
        self.trace_size[trace_key] = trace_length
        self.labels.update(labels)

        if log2_probability is None:
            self.total_number_non_fitting_traces += frequency
        else:
            self.log2_of_model_probability[trace_key] = log2_probability

    @staticmethod
    def h_0(accumulated_rho: int, total_number_of_traces: int) -> float:
        if accumulated_rho == 0 or accumulated_rho == total_number_of_traces:
            return 0.0
        p = accumulated_rho / total_number_of_traces
        return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

    def compute_relevance(self) -> float:
        accumulated_rho = 0
        accumulated_cost_bits = 0.0

        if self.number_of_traces <= 0:
            return float("nan")

        label_cost = math.log2(1 + len(self.labels)) if self.labels else 0.0

        for trace_key, trace_freq in self.trace_frequency.items():
            if trace_key in self.log2_of_model_probability:
                cost_bits = -self.log2_of_model_probability[trace_key]
                accumulated_rho += trace_freq
            else:
                cost_bits = (1 + self.trace_size[trace_key]) * label_cost

            accumulated_cost_bits += (cost_bits * trace_freq) / self.number_of_traces

        return self.h_0(accumulated_rho, self.number_of_traces) + accumulated_cost_bits
