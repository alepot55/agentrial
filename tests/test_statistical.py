"""Tests for statistical methods."""

import pytest

from agentrial.metrics.statistical import (
    benjamini_hochberg_correction,
    bootstrap_confidence_interval,
    compare_multiple_metrics,
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
        """Test CI with mixed results.

        For p=0.7, n=10 the Wilson 95% CI is approximately [0.397, 0.892].
        The lower bound must be ABOVE 0.39 (Wald gives ~0.35), proving
        Wilson's superior boundary behavior.
        """
        ci = wilson_confidence_interval(7, 10)
        assert 0.39 <= ci.lower <= 0.42  # Wilson-specific range
        assert 0.88 <= ci.upper <= 0.91  # Wilson-specific range

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
        """Test bootstrap with varied values.

        For uniform 1..10 data (mean=5.5), 95% bootstrap CI should be
        reasonably tight — roughly [3.5, 7.5]. Must NOT accept [0.01, 100].
        """
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        ci = bootstrap_confidence_interval(values, seed=42)
        assert 2.5 <= ci.lower <= 5.0  # Reasonably tight lower bound
        assert 6.0 <= ci.upper <= 8.5  # Reasonably tight upper bound

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
        """Test that odds ratio is returned.

        For 8/10 vs 2/10 the true odds ratio is (8*8)/(2*2) = 16.0.
        """
        odds_ratio, _ = fisher_exact_test(8, 10, 2, 10)
        assert odds_ratio > 5.0  # Must be well above 1 (actual = 16)


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


class TestBenjaminiHochbergCorrection:
    """Tests for Benjamini-Hochberg FDR correction."""

    def test_empty_pvalues(self) -> None:
        """Test with empty p-value list."""
        significant, adjusted = benjamini_hochberg_correction([])
        assert significant == []
        assert adjusted == []

    def test_single_pvalue_significant(self) -> None:
        """Test single significant p-value."""
        significant, adjusted = benjamini_hochberg_correction([0.01], alpha=0.05)
        assert significant == [True]
        assert adjusted[0] == pytest.approx(0.01)

    def test_single_pvalue_not_significant(self) -> None:
        """Test single non-significant p-value."""
        significant, adjusted = benjamini_hochberg_correction([0.10], alpha=0.05)
        assert significant == [False]
        assert adjusted[0] == pytest.approx(0.10)

    def test_multiple_all_significant(self) -> None:
        """Test multiple p-values all below threshold."""
        p_values = [0.001, 0.002, 0.003]
        significant, adjusted = benjamini_hochberg_correction(p_values, alpha=0.05)
        assert all(significant)
        # Adjusted p-values should be larger than originals
        for orig, adj in zip(p_values, adjusted, strict=True):
            assert adj >= orig

    def test_multiple_none_significant(self) -> None:
        """Test multiple p-values all above threshold."""
        p_values = [0.20, 0.30, 0.40]
        significant, adjusted = benjamini_hochberg_correction(p_values, alpha=0.05)
        assert not any(significant)

    def test_multiple_mixed(self) -> None:
        """Test with mixed significant/non-significant p-values."""
        # p1=0.01, p2=0.04, p3=0.10
        # Sorted: 0.01, 0.04, 0.10
        # Thresholds: 0.0167, 0.0333, 0.05
        # p1 < 0.0167: sig, p2 > 0.0333: not sig
        p_values = [0.01, 0.04, 0.10]
        significant, adjusted = benjamini_hochberg_correction(p_values, alpha=0.05)
        assert significant[0]  # 0.01 is significant
        assert not significant[2]  # 0.10 is not significant

    def test_adjusted_pvalues_monotonic(self) -> None:
        """Test that adjusted p-values are monotonic in sorted order."""
        p_values = [0.01, 0.03, 0.02, 0.05]
        significant, adjusted = benjamini_hochberg_correction(p_values, alpha=0.05)
        # When sorted by original p-value, adjusted should be monotonically increasing
        sorted_indices = sorted(range(len(p_values)), key=lambda i: p_values[i])
        sorted_adjusted = [adjusted[i] for i in sorted_indices]
        for i in range(len(sorted_adjusted) - 1):
            assert sorted_adjusted[i] <= sorted_adjusted[i + 1]


class TestCompareMultipleMetrics:
    """Tests for comparing multiple metrics with FDR correction."""

    def test_empty_comparisons(self) -> None:
        """Test with empty comparisons."""
        result = compare_multiple_metrics([])
        assert result["results"] == []
        assert not result["any_significant"]
        assert result["n_significant"] == 0

    def test_all_significant(self) -> None:
        """Test when all metrics show significant change."""
        comparisons = [
            {"name": "pass_rate", "p_value": 0.001},
            {"name": "cost", "p_value": 0.002},
            {"name": "latency", "p_value": 0.003},
        ]
        result = compare_multiple_metrics(comparisons, alpha=0.05)
        assert result["any_significant"]
        assert result["n_significant"] == 3
        # All should remain significant after correction
        for r in result["results"]:
            assert r["significant_after_correction"]

    def test_none_significant(self) -> None:
        """Test when no metrics show significant change."""
        comparisons = [
            {"name": "pass_rate", "p_value": 0.20},
            {"name": "cost", "p_value": 0.30},
            {"name": "latency", "p_value": 0.40},
        ]
        result = compare_multiple_metrics(comparisons, alpha=0.05)
        assert not result["any_significant"]
        assert result["n_significant"] == 0

    def test_some_significant(self) -> None:
        """Test with some significant, some not."""
        comparisons = [
            {"name": "pass_rate", "p_value": 0.001},  # Clearly significant
            {"name": "cost", "p_value": 0.50},  # Not significant
            {"name": "latency", "p_value": 0.60},  # Not significant
        ]
        result = compare_multiple_metrics(comparisons, alpha=0.05)
        assert result["any_significant"]
        assert result["n_significant"] >= 1

    def test_adjusted_pvalues_included(self) -> None:
        """Test that adjusted p-values are included in results."""
        comparisons = [
            {"name": "pass_rate", "p_value": 0.01},
        ]
        result = compare_multiple_metrics(comparisons, alpha=0.05)
        assert "adjusted_p_value" in result["results"][0]
        assert "significant_after_correction" in result["results"][0]
