"""Canary routing with automatic rollback (Phase 5).

Splits traffic between a stable model and a canary model. Tracks the canary's
recent error rate and latency in a rolling window; if either breaches a
threshold, the canary is automatically disabled and all traffic reverts to
stable until manually re-enabled.
"""

import logging
import random
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger("canary")


@dataclass
class CanaryRouter:
    canary_traffic_fraction: float = 0.05
    window_size: int = 20
    max_error_rate: float = 0.2
    max_p95_latency_ms: float = 500.0

    enabled: bool = field(default=True, init=False)
    _outcomes: "deque[bool]" = field(default_factory=lambda: deque(maxlen=20), init=False)
    _latencies_ms: "deque[float]" = field(default_factory=lambda: deque(maxlen=20), init=False)

    def __post_init__(self) -> None:
        self._outcomes = deque(maxlen=self.window_size)
        self._latencies_ms = deque(maxlen=self.window_size)

    def should_route_to_canary(self) -> bool:
        return self.enabled and random.random() < self.canary_traffic_fraction

    def record_canary_result(self, success: bool, latency_ms: float) -> None:
        self._outcomes.append(success)
        self._latencies_ms.append(latency_ms)
        self._check_thresholds()

    def _check_thresholds(self) -> None:
        if not self.enabled or len(self._outcomes) < self.window_size:
            return

        error_rate = 1 - (sum(self._outcomes) / len(self._outcomes))
        sorted_latencies = sorted(self._latencies_ms)
        p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]

        if error_rate > self.max_error_rate or p95_latency_ms > self.max_p95_latency_ms:
            self.enabled = False
            logger.warning(
                "canary rollback triggered: error_rate=%.2f (max %.2f), "
                "p95_latency_ms=%.1f (max %.1f) -- reverting all traffic to stable",
                error_rate, self.max_error_rate, p95_latency_ms, self.max_p95_latency_ms,
            )

    def reset(self) -> None:
        """Re-enable the canary and clear rolling stats, e.g. after fixing a bad deploy."""
        self.enabled = True
        self._outcomes.clear()
        self._latencies_ms.clear()
        logger.info("canary manually reset -- re-enabled with cleared stats")

    def status(self) -> dict:
        error_rate = None
        p95_latency_ms = None
        if self._outcomes:
            error_rate = 1 - (sum(self._outcomes) / len(self._outcomes))
            sorted_latencies = sorted(self._latencies_ms)
            p95_latency_ms = sorted_latencies[int(len(sorted_latencies) * 0.95)]

        return {
            "enabled": self.enabled,
            "traffic_fraction": self.canary_traffic_fraction,
            "recent_error_rate": error_rate,
            "recent_p95_latency_ms": p95_latency_ms,
            "samples": len(self._outcomes),
        }
