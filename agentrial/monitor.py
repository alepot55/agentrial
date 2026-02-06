"""Regression Monitoring for Production.

Implements drift detection algorithms to monitor agent performance
in production against a baseline snapshot.

Drift detection methods:
- CUSUM (Cumulative Sum): Sensitive to gradual shifts in pass rate
- Page-Hinkley: CUSUM variant robust to normal variations
- Kolmogorov-Smirnov test: Compares cost/latency distributions
- Sliding window + Fisher test: Pass rate with small samples

Usage:
    from agentrial.monitor import DriftDetector
    detector = DriftDetector(baseline_pass_rate=0.85)
    for observation in stream:
        alert = detector.observe(observation)
        if alert:
            print(f"DRIFT DETECTED: {alert}")
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DriftAlert:
    """Alert emitted when drift is detected."""

    method: str
    metric: str
    message: str
    p_value: float | None = None
    baseline_value: float = 0.0
    current_value: float = 0.0
    observation_count: int = 0
    timestamp: float = 0.0
    severity: str = "warning"

    @property
    def significant(self) -> bool:
        """Whether the drift is statistically significant."""
        if self.p_value is not None:
            return self.p_value < 0.05
        return True  # Non-statistical methods always flag


@dataclass
class Observation:
    """A single observation from production."""

    passed: bool
    cost: float = 0.0
    latency_ms: float = 0.0
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CUSUMDetector:
    """CUSUM (Cumulative Sum) drift detector.

    Sensitive to gradual shifts in a metric. Maintains cumulative
    sums that trigger when they exceed a threshold.
    """

    def __init__(
        self,
        target: float,
        threshold: float = 5.0,
        drift_magnitude: float = 0.1,
    ) -> None:
        """Initialize CUSUM detector.

        Args:
            target: Expected (baseline) value of the metric.
            threshold: Decision threshold h (higher = less sensitive).
            drift_magnitude: Minimum shift to detect (delta/2 in CUSUM).
        """
        self.target = target
        self.threshold = threshold
        self.allowance = drift_magnitude / 2  # k in CUSUM formula
        self.s_high = 0.0  # Upper CUSUM
        self.s_low = 0.0  # Lower CUSUM
        self.count = 0

    def update(self, value: float) -> DriftAlert | None:
        """Update with a new observation.

        Args:
            value: The observed metric value.

        Returns:
            DriftAlert if drift detected, None otherwise.
        """
        self.count += 1

        # Update cumulative sums
        self.s_high = max(0, self.s_high + (value - self.target) - self.allowance)
        self.s_low = max(0, self.s_low - (value - self.target) - self.allowance)

        if self.s_high > self.threshold:
            alert = DriftAlert(
                method="cusum",
                metric="pass_rate",
                message=(
                    f"CUSUM detected upward shift "
                    f"(S_high={self.s_high:.2f} > {self.threshold})"
                ),
                baseline_value=self.target,
                current_value=value,
                observation_count=self.count,
                timestamp=time.time(),
            )
            self.s_high = 0  # Reset after alert
            return alert

        if self.s_low > self.threshold:
            alert = DriftAlert(
                method="cusum",
                metric="pass_rate",
                message=(
                    f"CUSUM detected downward shift "
                    f"(S_low={self.s_low:.2f} > {self.threshold})"
                ),
                baseline_value=self.target,
                current_value=value,
                observation_count=self.count,
                timestamp=time.time(),
                severity="critical",
            )
            self.s_low = 0  # Reset after alert
            return alert

        return None

    def reset(self) -> None:
        """Reset the detector state."""
        self.s_high = 0.0
        self.s_low = 0.0
        self.count = 0


class PageHinkleyDetector:
    """Page-Hinkley drift detector.

    A CUSUM variant that is more robust to normal variations.
    Uses a running mean and detects when cumulative deviation
    exceeds a threshold.
    """

    def __init__(
        self,
        threshold: float = 50.0,
        alpha: float = 0.005,
        min_observations: int = 30,
    ) -> None:
        """Initialize Page-Hinkley detector.

        Args:
            threshold: Detection threshold (lambda).
            alpha: Tolerance for acceptable deviation (delta).
            min_observations: Minimum observations before detecting.
        """
        self.threshold = threshold
        self.alpha = alpha
        self.min_observations = min_observations
        self.count = 0
        self.sum = 0.0
        self.mean = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0

    def update(self, value: float) -> DriftAlert | None:
        """Update with a new observation.

        Args:
            value: The observed value.

        Returns:
            DriftAlert if drift detected, None otherwise.
        """
        self.count += 1
        self.sum += value
        self.mean = self.sum / self.count

        self.cumulative_sum += value - self.mean - self.alpha
        self.min_cumulative_sum = min(self.min_cumulative_sum, self.cumulative_sum)

        if self.count < self.min_observations:
            return None

        ph_value = self.cumulative_sum - self.min_cumulative_sum

        if ph_value > self.threshold:
            alert = DriftAlert(
                method="page_hinkley",
                metric="value",
                message=(
                    f"Page-Hinkley detected drift "
                    f"(PH={ph_value:.2f} > {self.threshold})"
                ),
                current_value=value,
                baseline_value=self.mean,
                observation_count=self.count,
                timestamp=time.time(),
            )
            self.reset()
            return alert

        return None

    def reset(self) -> None:
        """Reset the detector state."""
        self.count = 0
        self.sum = 0.0
        self.mean = 0.0
        self.cumulative_sum = 0.0
        self.min_cumulative_sum = 0.0


def ks_two_sample(
    sample_a: list[float],
    sample_b: list[float],
) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test.

    Tests whether two samples come from the same distribution.

    Args:
        sample_a: First sample.
        sample_b: Second sample.

    Returns:
        Tuple of (KS statistic, approximate p-value).
    """
    if not sample_a or not sample_b:
        return 0.0, 1.0

    n_a = len(sample_a)
    n_b = len(sample_b)

    # Combine and sort
    all_values = sorted(set(sample_a + sample_b))

    # Compute empirical CDFs
    d_max = 0.0
    for val in all_values:
        cdf_a = sum(1 for x in sample_a if x <= val) / n_a
        cdf_b = sum(1 for x in sample_b if x <= val) / n_b
        d_max = max(d_max, abs(cdf_a - cdf_b))

    # Approximate p-value using asymptotic formula
    n_eff = (n_a * n_b) / (n_a + n_b)
    lambda_val = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d_max

    # Kolmogorov distribution approximation
    if lambda_val <= 0:
        p_value = 1.0
    elif lambda_val < 0.27:
        p_value = 1.0
    elif lambda_val > 3.1:
        p_value = 0.0
    else:
        # Series approximation
        p_value = 0.0
        for k in range(1, 100):
            term = (-1) ** (k - 1) * math.exp(-2 * k * k * lambda_val * lambda_val)
            p_value += term
        p_value = 2.0 * p_value
        p_value = max(0.0, min(1.0, p_value))

    return d_max, p_value


class SlidingWindowDetector:
    """Sliding window + Fisher exact test for pass rate monitoring.

    Compares pass rate in a recent window against the baseline using
    Fisher's exact test equivalent (chi-squared for larger samples).
    """

    def __init__(
        self,
        baseline_pass_rate: float,
        baseline_n: int = 100,
        window_size: int = 50,
        alpha: float = 0.05,
    ) -> None:
        """Initialize sliding window detector.

        Args:
            baseline_pass_rate: Expected pass rate from baseline.
            baseline_n: Number of observations in baseline.
            window_size: Size of the sliding window.
            alpha: Significance level for detection.
        """
        self.baseline_pass_rate = baseline_pass_rate
        self.baseline_n = baseline_n
        self.window_size = window_size
        self.alpha = alpha
        self._window: list[bool] = []

    def update(self, passed: bool) -> DriftAlert | None:
        """Update with a new pass/fail observation.

        Args:
            passed: Whether the observation passed.

        Returns:
            DriftAlert if significant drift detected.
        """
        self._window.append(passed)

        # Keep window at max size
        if len(self._window) > self.window_size:
            self._window = self._window[-self.window_size :]

        # Need full window before testing
        if len(self._window) < self.window_size:
            return None

        # Compute window pass rate
        window_passes = sum(self._window)
        window_rate = window_passes / self.window_size

        # Use normal approximation for proportion test
        p_value = self._proportion_test(
            window_passes,
            self.window_size,
            self.baseline_pass_rate,
        )

        if p_value < self.alpha:
            severity = "critical" if window_rate < self.baseline_pass_rate else "info"
            return DriftAlert(
                method="sliding_window",
                metric="pass_rate",
                message=(
                    f"Pass rate drift: {window_rate:.0%} vs "
                    f"baseline {self.baseline_pass_rate:.0%} "
                    f"(p={p_value:.4f})"
                ),
                p_value=p_value,
                baseline_value=self.baseline_pass_rate,
                current_value=window_rate,
                observation_count=len(self._window),
                timestamp=time.time(),
                severity=severity,
            )

        return None

    def _proportion_test(
        self,
        successes: int,
        n: int,
        expected_rate: float,
    ) -> float:
        """Two-proportion z-test approximation.

        Args:
            successes: Number of successes in window.
            n: Window size.
            expected_rate: Expected proportion.

        Returns:
            Approximate p-value.
        """
        if n == 0:
            return 1.0

        observed_rate = successes / n

        # Pooled proportion
        pooled = (
            (successes + expected_rate * self.baseline_n)
            / (n + self.baseline_n)
        )

        # Avoid division by zero
        if pooled <= 0 or pooled >= 1:
            return 1.0

        # Standard error
        se = math.sqrt(pooled * (1 - pooled) * (1 / n + 1 / self.baseline_n))

        if se == 0:
            return 1.0

        # Z statistic
        z = (observed_rate - expected_rate) / se

        # Two-tailed p-value from normal CDF approximation
        p_value = 2 * (1 - _normal_cdf(abs(z)))
        return p_value

    def reset(self) -> None:
        """Reset the window."""
        self._window = []


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using Abramowitz & Stegun."""
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0

    # Constants for approximation
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    t = 1.0 / (1.0 + b0 * abs(x))
    pdf = math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    cdf = 1.0 - pdf * poly

    return cdf if x >= 0 else 1.0 - cdf


class DriftDetector:
    """Unified drift detector combining multiple methods.

    Monitors pass rate, cost, and latency using appropriate
    detection methods for each metric type.
    """

    def __init__(
        self,
        baseline_pass_rate: float = 0.85,
        baseline_costs: list[float] | None = None,
        baseline_latencies: list[float] | None = None,
        cusum_threshold: float = 5.0,
        window_size: int = 50,
    ) -> None:
        """Initialize the unified detector.

        Args:
            baseline_pass_rate: Expected pass rate.
            baseline_costs: Baseline cost distribution.
            baseline_latencies: Baseline latency distribution.
            cusum_threshold: CUSUM detection threshold.
            window_size: Sliding window size.
        """
        self.baseline_pass_rate = baseline_pass_rate
        self.baseline_costs = baseline_costs or []
        self.baseline_latencies = baseline_latencies or []

        # Initialize sub-detectors
        self.cusum = CUSUMDetector(
            target=baseline_pass_rate,
            threshold=cusum_threshold,
        )
        self.page_hinkley = PageHinkleyDetector()
        self.sliding_window = SlidingWindowDetector(
            baseline_pass_rate=baseline_pass_rate,
            window_size=window_size,
        )

        # Collect recent cost/latency for KS test
        self._recent_costs: list[float] = []
        self._recent_latencies: list[float] = []
        self._ks_window = window_size
        self._observations = 0

    def observe(self, obs: Observation) -> list[DriftAlert]:
        """Process a new observation and check for drift.

        Args:
            obs: The production observation.

        Returns:
            List of drift alerts (empty if no drift detected).
        """
        self._observations += 1
        alerts: list[DriftAlert] = []

        # Pass rate monitoring
        value = 1.0 if obs.passed else 0.0

        cusum_alert = self.cusum.update(value)
        if cusum_alert:
            alerts.append(cusum_alert)

        window_alert = self.sliding_window.update(obs.passed)
        if window_alert:
            alerts.append(window_alert)

        # Cost distribution monitoring
        if obs.cost > 0:
            self._recent_costs.append(obs.cost)
            if len(self._recent_costs) > self._ks_window:
                self._recent_costs = self._recent_costs[-self._ks_window :]

            if (
                self.baseline_costs
                and len(self._recent_costs) >= self._ks_window
            ):
                ks_stat, ks_p = ks_two_sample(
                    self.baseline_costs, self._recent_costs
                )
                if ks_p < 0.05:
                    alerts.append(
                        DriftAlert(
                            method="ks_test",
                            metric="cost",
                            message=(
                                f"Cost distribution drift "
                                f"(KS={ks_stat:.3f}, p={ks_p:.4f})"
                            ),
                            p_value=ks_p,
                            observation_count=self._observations,
                            timestamp=time.time(),
                        )
                    )

        # Latency distribution monitoring
        if obs.latency_ms > 0:
            self._recent_latencies.append(obs.latency_ms)
            if len(self._recent_latencies) > self._ks_window:
                self._recent_latencies = self._recent_latencies[
                    -self._ks_window :
                ]

            if (
                self.baseline_latencies
                and len(self._recent_latencies) >= self._ks_window
            ):
                ks_stat, ks_p = ks_two_sample(
                    self.baseline_latencies, self._recent_latencies
                )
                if ks_p < 0.05:
                    alerts.append(
                        DriftAlert(
                            method="ks_test",
                            metric="latency",
                            message=(
                                f"Latency distribution drift "
                                f"(KS={ks_stat:.3f}, p={ks_p:.4f})"
                            ),
                            p_value=ks_p,
                            observation_count=self._observations,
                            timestamp=time.time(),
                        )
                    )

        return alerts

    def reset(self) -> None:
        """Reset all detectors."""
        self.cusum.reset()
        self.page_hinkley.reset()
        self.sliding_window.reset()
        self._recent_costs = []
        self._recent_latencies = []
        self._observations = 0

    @property
    def observation_count(self) -> int:
        """Total observations processed."""
        return self._observations


def print_alert(
    alert: DriftAlert,
    console: Any = None,
) -> None:
    """Print a drift alert to terminal.

    Args:
        alert: The alert to display.
        console: Rich console (creates one if not provided).
    """
    from rich.console import Console

    if console is None:
        console = Console()

    severity_style = {
        "critical": "bold red",
        "warning": "yellow",
        "info": "blue",
    }

    style = severity_style.get(alert.severity, "white")
    icon = {"critical": "!!!", "warning": "!!", "info": "i"}.get(
        alert.severity, "?"
    )

    console.print(
        f"[{style}][{icon}] DRIFT [{alert.method}] {alert.metric}: "
        f"{alert.message}[/{style}]"
    )
