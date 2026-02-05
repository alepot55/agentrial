"""Tests for statistical methods."""

import pytest

from agenteval.metrics.statistical import (
    bootstrap_confidence_interval,
    detect_regression,
    fisher_exact_test,
    mann_whitney_u_test,
    wilson_confidence_interval,
)


class TestWilsonConfidenceInterval:
    """Tests for Wilson score confidence interval."""

    def test_all_success(self) -> None:
        """Test CI when all trials succeed."""
        ci = wilson_confidence_interval(10, 10)
        assert ci.lower > 0.7
        assert ci.upper == pytest.approx(1.0, rel=1e-6)
        assert ci.confidence_level == 0.95

    def test_all_failure(self) -> None:
        """Test CI when all trials fail."""
        ci = wilson_confidence_interval(0, 10)
        assert ci.lower == 0.0
        assert ci.upper < 0.3
        assert ci.confidence_level == 0.95

    def test_mixed_results(self) -> None:
        """Test CI with mixed results."""
        ci = wilson_confidence_interval(7, 10)
        assert 0.35 < ci.lower < 0.7  # Wilson CI for 70% with n=10
        assert 0.8 < ci.upper < 1.0

    def test_empty_trials(self) -> None:
        """Test CI with no trials."""
        ci = wilson_confidence_interval(0, 0)
        assert ci.lower == 0.0
        assert ci.upper == 1.0

    def test_different_confidence_level(self) -> None:
        """Test CI with different confidence level."""
        ci_95 = wilson_confidence_interval(7, 10, 0.95)
        ci_99 = wilson_confidence_interval(7, 10, 0.99)
        # 99% CI should be wider
        assert ci_99.upper - ci_99.lower > ci_95.upper - ci_95.lower


class TestBootstrapConfidenceInterval:
    """Tests for bootstrap confidence interval."""

    def test_single_value(self) -> None:
        """Test bootstrap with single value."""
        ci = bootstrap_confidence_interval([5.0])
        assert ci.lower == 5.0
        assert ci.upper == 5.0

    def test_identical_values(self) -> None:
        """Test bootstrap with identical values."""
        ci = bootstrap_confidence_interval([5.0, 5.0, 5.0, 5.0, 5.0])
        assert ci.lower == 5.0
        assert ci.upper == 5.0

    def test_varied_values(self) -> None:
        """Test bootstrap with varied values."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ci = bootstrap_confidence_interval(values, seed=42)
        assert ci.lower < 5.5
        assert ci.upper > 5.5
        assert ci.lower > 0

    def test_empty_values(self) -> None:
        """Test bootstrap with empty list."""
        ci = bootstrap_confidence_interval([])
        assert ci.lower == 0.0
        assert ci.upper == 0.0

    def test_median_statistic(self) -> None:
        """Test bootstrap with median statistic."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0]  # Skewed
        ci_mean = bootstrap_confidence_interval(values, statistic="mean", seed=42)
        ci_median = bootstrap_confidence_interval(values, statistic="median", seed=42)
        # Median CI center should be lower than mean due to outlier
        median_center = (ci_median.lower + ci_median.upper) / 2
        mean_center = (ci_mean.lower + ci_mean.upper) / 2
        assert median_center < mean_center


class TestFisherExactTest:
    """Tests for Fisher's exact test."""

    def test_identical_proportions(self) -> None:
        """Test with identical proportions."""
        _, p = fisher_exact_test(7, 10, 7, 10)
        assert p == 1.0  # No difference

    def test_different_proportions(self) -> None:
        """Test with different proportions."""
        _, p = fisher_exact_test(9, 10, 1, 10)
        assert p < 0.01  # Significant difference

    def test_returns_odds_ratio(self) -> None:
        """Test that odds ratio is returned."""
        odds_ratio, _ = fisher_exact_test(8, 10, 2, 10)
        assert odds_ratio > 1.0  # Group A has higher success rate


class TestMannWhitneyU:
    """Tests for Mann-Whitney U test."""

    def test_identical_distributions(self) -> None:
        """Test with identical distributions."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        _, p = mann_whitney_u_test(values, values)
        assert p > 0.05

    def test_different_distributions(self) -> None:
        """Test with clearly different distributions."""
        a = [1.0, 2.0, 3.0, 4.0, 5.0]
        b = [10.0, 11.0, 12.0, 13.0, 14.0]
        _, p = mann_whitney_u_test(a, b)
        assert p < 0.05

    def test_empty_list(self) -> None:
        """Test with empty list."""
        _, p = mann_whitney_u_test([], [1.0, 2.0, 3.0])
        assert p == 1.0


class TestDetectRegression:
    """Tests for regression detection."""

    def test_no_regression(self) -> None:
        """Test when there's no regression."""
        result = detect_regression(8, 10, 7, 10)
        assert not result["is_regression"]

    def test_clear_regression(self) -> None:
        """Test when there's a clear regression."""
        result = detect_regression(2, 10, 9, 10)
        assert result["is_regression"]
        assert result["rate_delta"] < 0

    def test_improvement_not_regression(self) -> None:
        """Test that improvement is not flagged as regression."""
        result = detect_regression(9, 10, 2, 10)
        assert not result["is_regression"]
        assert result["rate_delta"] > 0
