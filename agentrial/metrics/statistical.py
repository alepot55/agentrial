"""Statistical methods for AgentEval.

This module implements statistically rigorous methods for:
- Confidence intervals for proportions (Wilson score)
- Bootstrap confidence intervals for continuous metrics
- Hypothesis tests for regression detection
"""

import numpy as np
from scipy import stats

from agentrial.types import ConfidenceInterval


def wilson_confidence_interval(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Compute Wilson score confidence interval for a proportion.

    The Wilson score interval is more accurate than the normal approximation
    for small samples and extreme proportions (near 0% or 100%).

    Args:
        successes: Number of successes.
        trials: Total number of trials.
        confidence_level: Confidence level (default 0.95 for 95% CI).

    Returns:
        ConfidenceInterval with lower and upper bounds.

    Reference:
        Wilson, E.B. (1927). "Probable Inference, the Law of Succession,
        and Statistical Inference"
    """
    if trials == 0:
        return ConfidenceInterval(lower=0.0, upper=1.0, confidence_level=confidence_level)

    p = successes / trials
    z = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    z2 = z * z

    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    margin = (z / denominator) * np.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return ConfidenceInterval(lower=lower, upper=upper, confidence_level=confidence_level)


def bootstrap_confidence_interval(
    values: list[float],
    confidence_level: float = 0.95,
    n_iterations: int = 500,
    statistic: str = "mean",
    seed: int | None = None,
) -> ConfidenceInterval:
    """Compute bootstrap confidence interval for a metric.

    Bootstrap resampling provides non-parametric confidence intervals
    that work well for non-normal distributions.

    Args:
        values: List of observed values.
        confidence_level: Confidence level (default 0.95).
        n_iterations: Number of bootstrap iterations (default 500).
        statistic: Which statistic to compute ("mean" or "median").
        seed: Random seed for reproducibility.

    Returns:
        ConfidenceInterval with lower and upper bounds.
    """
    if not values:
        return ConfidenceInterval(lower=0.0, upper=0.0, confidence_level=confidence_level)

    if len(values) == 1:
        return ConfidenceInterval(
            lower=values[0], upper=values[0], confidence_level=confidence_level
        )

    rng = np.random.default_rng(seed)
    values_array = np.array(values)

    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_iterations):
        sample = rng.choice(values_array, size=len(values_array), replace=True)
        if statistic == "mean":
            bootstrap_stats.append(np.mean(sample))
        elif statistic == "median":
            bootstrap_stats.append(np.median(sample))
        else:
            raise ValueError(f"Unknown statistic: {statistic}")

    # Compute percentile interval
    alpha = 1 - confidence_level
    lower_percentile = alpha / 2 * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower = float(np.percentile(bootstrap_stats, lower_percentile))
    upper = float(np.percentile(bootstrap_stats, upper_percentile))

    return ConfidenceInterval(lower=lower, upper=upper, confidence_level=confidence_level)


def fisher_exact_test(
    successes_a: int,
    trials_a: int,
    successes_b: int,
    trials_b: int,
) -> tuple[float, float]:
    """Perform Fisher's exact test for comparing two proportions.

    This test is appropriate for small samples where chi-squared
    approximation is unreliable.

    Args:
        successes_a: Successes in group A.
        trials_a: Total trials in group A.
        successes_b: Successes in group B.
        trials_b: Total trials in group B.

    Returns:
        Tuple of (odds_ratio, p_value).
    """
    # Build 2x2 contingency table
    table = [
        [successes_a, trials_a - successes_a],
        [successes_b, trials_b - successes_b],
    ]

    odds_ratio, p_value = stats.fisher_exact(table)
    return float(odds_ratio), float(p_value)


def mann_whitney_u_test(
    values_a: list[float],
    values_b: list[float],
) -> tuple[float, float]:
    """Perform Mann-Whitney U test for comparing two distributions.

    This non-parametric test compares whether values in one group
    tend to be larger than in another, without assuming normality.

    Args:
        values_a: Values from group A.
        values_b: Values from group B.

    Returns:
        Tuple of (U_statistic, p_value).
    """
    if not values_a or not values_b:
        return 0.0, 1.0

    statistic, p_value = stats.mannwhitneyu(
        values_a,
        values_b,
        alternative="two-sided",
    )

    return float(statistic), float(p_value)


def detect_regression(
    current_successes: int,
    current_trials: int,
    baseline_successes: int,
    baseline_trials: int,
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Detect if there's a statistically significant regression.

    Args:
        current_successes: Successes in current run.
        current_trials: Total trials in current run.
        baseline_successes: Successes in baseline.
        baseline_trials: Total trials in baseline.
        alpha: Significance level (default 0.05).

    Returns:
        Dictionary with:
        - is_regression: Whether a regression was detected
        - p_value: P-value from Fisher's exact test
        - current_rate: Current pass rate
        - baseline_rate: Baseline pass rate
        - rate_delta: Change in pass rate
    """
    _, p_value = fisher_exact_test(
        current_successes,
        current_trials,
        baseline_successes,
        baseline_trials,
    )

    current_rate = current_successes / current_trials if current_trials > 0 else 0
    baseline_rate = baseline_successes / baseline_trials if baseline_trials > 0 else 0

    # Regression is when current is significantly worse than baseline
    is_regression = p_value < alpha and current_rate < baseline_rate

    return {
        "is_regression": is_regression,
        "p_value": p_value,
        "current_rate": current_rate,
        "baseline_rate": baseline_rate,
        "rate_delta": current_rate - baseline_rate,
    }


def compare_distributions(
    current_values: list[float],
    baseline_values: list[float],
    alpha: float = 0.05,
) -> dict[str, float | bool]:
    """Compare two distributions for significant difference.

    Uses Mann-Whitney U test for non-parametric comparison.

    Args:
        current_values: Values from current run.
        baseline_values: Values from baseline.
        alpha: Significance level.

    Returns:
        Dictionary with:
        - is_different: Whether distributions differ significantly
        - p_value: P-value from Mann-Whitney U test
        - current_median: Median of current values
        - baseline_median: Median of baseline values
        - median_delta: Change in median
    """
    if not current_values or not baseline_values:
        return {
            "is_different": False,
            "p_value": 1.0,
            "current_median": 0.0,
            "baseline_median": 0.0,
            "median_delta": 0.0,
        }

    _, p_value = mann_whitney_u_test(current_values, baseline_values)

    current_median = float(np.median(current_values))
    baseline_median = float(np.median(baseline_values))

    return {
        "is_different": p_value < alpha,
        "p_value": p_value,
        "current_median": current_median,
        "baseline_median": baseline_median,
        "median_delta": current_median - baseline_median,
    }


def benjamini_hochberg_correction(
    p_values: list[float],
    alpha: float = 0.05,
) -> tuple[list[bool], list[float]]:
    """Apply Benjamini-Hochberg FDR correction for multiple testing.

    When comparing multiple metrics simultaneously (e.g., pass rate, cost,
    latency), the probability of false positives increases. This correction
    controls the False Discovery Rate (FDR) - the expected proportion of
    false positives among all rejected hypotheses.

    Args:
        p_values: List of p-values from individual tests.
        alpha: Desired FDR level (default 0.05).

    Returns:
        Tuple of:
        - List of booleans indicating which tests are significant
        - List of adjusted p-values

    Reference:
        Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery
        rate: a practical and powerful approach to multiple testing.
    """
    if not p_values:
        return [], []

    n = len(p_values)

    # Create (index, p_value) pairs and sort by p_value
    indexed_pvalues = sorted(enumerate(p_values), key=lambda x: x[1])

    # Calculate adjusted p-values
    adjusted = [0.0] * n
    significant = [False] * n

    # Process from largest to smallest p-value
    min_so_far = 1.0
    for rank, (original_idx, p) in enumerate(reversed(indexed_pvalues)):
        # Adjusted p = p * n / (n - rank) = p * n / actual_rank
        actual_rank = n - rank
        adj_p = min(p * n / actual_rank, 1.0)
        # Ensure monotonicity
        adj_p = min(adj_p, min_so_far)
        min_so_far = adj_p
        adjusted[original_idx] = adj_p

    # Determine significance
    # Find largest rank k where p_(k) <= k/n * alpha
    max_significant_rank = 0
    for rank, (_original_idx, p) in enumerate(indexed_pvalues, 1):
        threshold = rank / n * alpha
        if p <= threshold:
            max_significant_rank = rank

    # Mark all tests with rank <= max_significant_rank as significant
    for rank, (original_idx, _) in enumerate(indexed_pvalues, 1):
        if rank <= max_significant_rank:
            significant[original_idx] = True

    return significant, adjusted


def compare_multiple_metrics(
    comparisons: list[dict[str, float]],
    alpha: float = 0.05,
) -> dict[str, any]:
    """Compare multiple metrics with FDR correction.

    When comparing current run to baseline across multiple metrics
    (pass rate, cost, latency), this function applies Benjamini-Hochberg
    correction to control the false discovery rate.

    Args:
        comparisons: List of dicts, each with 'name' and 'p_value' keys.
        alpha: Desired FDR level.

    Returns:
        Dictionary with:
        - results: List of comparison results with adjusted p-values
        - any_significant: Whether any metric shows significant change
        - n_significant: Number of significant changes
    """
    if not comparisons:
        return {
            "results": [],
            "any_significant": False,
            "n_significant": 0,
        }

    p_values = [c["p_value"] for c in comparisons]
    significant, adjusted = benjamini_hochberg_correction(p_values, alpha)

    results = []
    for i, comp in enumerate(comparisons):
        results.append({
            **comp,
            "adjusted_p_value": adjusted[i],
            "significant_after_correction": significant[i],
        })

    return {
        "results": results,
        "any_significant": any(significant),
        "n_significant": sum(significant),
    }
