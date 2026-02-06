"""Tests for Regression Monitoring."""

from __future__ import annotations

from agentrial.monitor import (
    CUSUMDetector,
    DriftAlert,
    DriftDetector,
    Observation,
    PageHinkleyDetector,
    SlidingWindowDetector,
    _normal_cdf,
    ks_two_sample,
)


class TestNormalCDF:
    """Tests for normal CDF approximation."""

    def test_zero(self) -> None:
        assert abs(_normal_cdf(0) - 0.5) < 0.001

    def test_positive(self) -> None:
        # CDF(1.96) ≈ 0.975
        assert abs(_normal_cdf(1.96) - 0.975) < 0.01

    def test_negative(self) -> None:
        # CDF(-1.96) ≈ 0.025
        assert abs(_normal_cdf(-1.96) - 0.025) < 0.01

    def test_large_positive(self) -> None:
        assert _normal_cdf(10) == 1.0

    def test_large_negative(self) -> None:
        assert _normal_cdf(-10) == 0.0


class TestKSTwoSample:
    """Tests for Kolmogorov-Smirnov test."""

    def test_identical_samples(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        ks_stat, p_value = ks_two_sample(a, a)
        assert ks_stat == 0.0
        assert p_value == 1.0

    def test_different_samples(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        b = [10.0, 20.0, 30.0, 40.0, 50.0] * 10
        ks_stat, p_value = ks_two_sample(a, b)
        assert ks_stat > 0.5
        assert p_value < 0.05

    def test_empty_samples(self) -> None:
        ks_stat, p_value = ks_two_sample([], [1.0, 2.0])
        assert ks_stat == 0.0
        assert p_value == 1.0

    def test_similar_samples(self) -> None:
        import random

        random.seed(42)
        a = [random.gauss(10, 1) for _ in range(50)]
        b = [random.gauss(10, 1) for _ in range(50)]
        _, p_value = ks_two_sample(a, b)
        # Similar distributions should have high p-value
        assert p_value > 0.05


class TestCUSUMDetector:
    """Tests for CUSUM drift detector."""

    def test_no_drift(self) -> None:
        detector = CUSUMDetector(target=0.85, threshold=5.0)
        # Feed values around the target
        for _ in range(100):
            alert = detector.update(0.85)
            assert alert is None

    def test_downward_drift(self) -> None:
        detector = CUSUMDetector(target=0.85, threshold=3.0, drift_magnitude=0.2)
        alerts = []
        # Feed consistently low values
        for _ in range(100):
            alert = detector.update(0.0)
            if alert:
                alerts.append(alert)

        assert len(alerts) > 0
        assert alerts[0].method == "cusum"
        assert "downward" in alerts[0].message

    def test_upward_drift(self) -> None:
        detector = CUSUMDetector(target=0.5, threshold=3.0, drift_magnitude=0.2)
        alerts = []
        for _ in range(100):
            alert = detector.update(1.0)
            if alert:
                alerts.append(alert)

        assert len(alerts) > 0
        assert "upward" in alerts[0].message

    def test_reset(self) -> None:
        detector = CUSUMDetector(target=0.85)
        detector.update(0.0)
        detector.update(0.0)
        assert detector.count == 2

        detector.reset()
        assert detector.count == 0
        assert detector.s_high == 0.0
        assert detector.s_low == 0.0


class TestPageHinkleyDetector:
    """Tests for Page-Hinkley detector."""

    def test_no_drift_stable(self) -> None:
        detector = PageHinkleyDetector(
            threshold=50.0, min_observations=10
        )
        for _ in range(20):
            alert = detector.update(1.0)
        # Stable signal should not trigger
        assert alert is None

    def test_drift_detected(self) -> None:
        detector = PageHinkleyDetector(
            threshold=10.0, alpha=0.005, min_observations=5
        )
        # Feed stable values first
        for _ in range(10):
            detector.update(0.0)

        # Then a significant shift
        alerts = []
        for _ in range(50):
            alert = detector.update(5.0)
            if alert:
                alerts.append(alert)

        assert len(alerts) > 0
        assert alerts[0].method == "page_hinkley"

    def test_min_observations_respected(self) -> None:
        detector = PageHinkleyDetector(min_observations=30)
        for _i in range(29):
            alert = detector.update(100.0)  # Extreme value
            assert alert is None  # Should not trigger before min

    def test_reset(self) -> None:
        detector = PageHinkleyDetector()
        detector.update(1.0)
        detector.update(2.0)
        assert detector.count == 2

        detector.reset()
        assert detector.count == 0


class TestSlidingWindowDetector:
    """Tests for sliding window + proportion test."""

    def test_no_drift(self) -> None:
        detector = SlidingWindowDetector(
            baseline_pass_rate=0.8, window_size=20
        )
        # 80% pass rate ≈ baseline
        for i in range(20):
            passed = i % 5 != 0  # 80%
            alert = detector.update(passed)

        # Should not trigger
        assert alert is None

    def test_regression_detected(self) -> None:
        detector = SlidingWindowDetector(
            baseline_pass_rate=0.9,
            baseline_n=100,
            window_size=20,
            alpha=0.05,
        )
        # All failures → 0% pass rate
        alerts = []
        for _ in range(25):
            alert = detector.update(False)
            if alert:
                alerts.append(alert)

        assert len(alerts) > 0
        assert alerts[0].method == "sliding_window"
        assert alerts[0].severity == "critical"

    def test_window_not_full(self) -> None:
        detector = SlidingWindowDetector(
            baseline_pass_rate=0.8, window_size=50
        )
        # Only 10 observations — should never trigger
        for _ in range(10):
            alert = detector.update(False)
            assert alert is None

    def test_reset(self) -> None:
        detector = SlidingWindowDetector(
            baseline_pass_rate=0.8, window_size=10
        )
        for _ in range(10):
            detector.update(True)

        detector.reset()
        assert len(detector._window) == 0


class TestDriftDetector:
    """Tests for unified DriftDetector."""

    def test_stable_observations(self) -> None:
        detector = DriftDetector(
            baseline_pass_rate=0.8,
            window_size=20,
        )
        for i in range(20):
            obs = Observation(passed=(i % 5 != 0), cost=0.01, latency_ms=100)
            detector.observe(obs)

        # Stable observations should not trigger drift
        # (some detectors may not trigger)
        assert detector.observation_count == 20

    def test_pass_rate_regression(self) -> None:
        detector = DriftDetector(
            baseline_pass_rate=0.9,
            cusum_threshold=2.0,
            window_size=20,
        )

        all_alerts = []
        for _ in range(30):
            obs = Observation(passed=False)
            alerts = detector.observe(obs)
            all_alerts.extend(alerts)

        # Should detect regression via CUSUM or sliding window
        assert len(all_alerts) > 0
        methods = {a.method for a in all_alerts}
        assert "cusum" in methods or "sliding_window" in methods

    def test_cost_drift(self) -> None:
        baseline_costs = [0.01] * 50
        detector = DriftDetector(
            baseline_pass_rate=1.0,
            baseline_costs=baseline_costs,
            window_size=50,
        )

        all_alerts = []
        for _ in range(60):
            obs = Observation(passed=True, cost=1.0)  # 100x increase
            alerts = detector.observe(obs)
            all_alerts.extend(alerts)

        cost_alerts = [a for a in all_alerts if a.metric == "cost"]
        assert len(cost_alerts) > 0

    def test_latency_drift(self) -> None:
        baseline_latencies = [100.0] * 50
        detector = DriftDetector(
            baseline_pass_rate=1.0,
            baseline_latencies=baseline_latencies,
            window_size=50,
        )

        all_alerts = []
        for _ in range(60):
            obs = Observation(passed=True, latency_ms=10000.0)  # 100x
            alerts = detector.observe(obs)
            all_alerts.extend(alerts)

        latency_alerts = [a for a in all_alerts if a.metric == "latency"]
        assert len(latency_alerts) > 0

    def test_reset(self) -> None:
        detector = DriftDetector(baseline_pass_rate=0.8)
        for _ in range(10):
            detector.observe(Observation(passed=True))

        assert detector.observation_count == 10
        detector.reset()
        assert detector.observation_count == 0


class TestDriftAlert:
    """Tests for DriftAlert dataclass."""

    def test_significant_with_p_value(self) -> None:
        alert = DriftAlert(
            method="test", metric="rate", message="drift", p_value=0.01
        )
        assert alert.significant

    def test_not_significant(self) -> None:
        alert = DriftAlert(
            method="test", metric="rate", message="ok", p_value=0.5
        )
        assert not alert.significant

    def test_no_p_value_defaults_significant(self) -> None:
        alert = DriftAlert(method="cusum", metric="rate", message="drift")
        assert alert.significant  # Non-statistical → always flags
