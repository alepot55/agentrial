"""Trajectory analysis and failure attribution for Agentrial.

This module implements step-level analysis to identify which trajectory
steps are most likely responsible for test failures.
"""

from collections import Counter
from typing import Any

from scipy import stats

from agentrial.metrics.statistical import benjamini_hochberg_correction
from agentrial.types import TrialResult


def compute_divergence(
    successful_trials: list[TrialResult],
    failed_trials: list[TrialResult],
    step_index: int,
) -> dict[str, Any]:
    """Compute divergence between successful and failed trials at a step.

    Uses Fisher's exact test to determine if the action distribution
    at this step differs significantly between successful and failed runs.

    Args:
        successful_trials: Trials that passed.
        failed_trials: Trials that failed.
        step_index: Which step to analyze.

    Returns:
        Dictionary with:
        - step_index: The analyzed step
        - p_value: P-value from Fisher's exact test
        - successful_actions: Action distribution in successful trials
        - failed_actions: Action distribution in failed trials
        - most_common_success: Most common action in successful trials
        - most_common_fail: Most common action in failed trials
    """
    # Extract actions at this step
    def get_action(trial: TrialResult) -> str | None:
        if step_index < len(trial.agent_output.steps):
            step = trial.agent_output.steps[step_index]
            return f"{step.step_type.value}:{step.name}"
        return None

    successful_actions = [get_action(t) for t in successful_trials if get_action(t)]
    failed_actions = [get_action(t) for t in failed_trials if get_action(t)]

    if not successful_actions and not failed_actions:
        return {
            "step_index": step_index,
            "p_value": 1.0,
            "successful_actions": {},
            "failed_actions": {},
            "most_common_success": None,
            "most_common_fail": None,
        }

    # Count action distributions
    success_counts = Counter(successful_actions)
    fail_counts = Counter(failed_actions)

    # Get all unique actions
    all_actions = set(success_counts.keys()) | set(fail_counts.keys())

    if len(all_actions) <= 1:
        # All same action, no divergence possible
        return {
            "step_index": step_index,
            "p_value": 1.0,
            "successful_actions": dict(success_counts),
            "failed_actions": dict(fail_counts),
            "most_common_success": success_counts.most_common(1)[0][0]
            if success_counts
            else None,
            "most_common_fail": fail_counts.most_common(1)[0][0] if fail_counts else None,
        }

    # Build contingency table for most common vs other actions
    most_common = max(all_actions, key=lambda a: success_counts.get(a, 0) + fail_counts.get(a, 0))

    table = [
        [
            success_counts.get(most_common, 0),
            sum(c for a, c in success_counts.items() if a != most_common),
        ],
        [
            fail_counts.get(most_common, 0),
            sum(c for a, c in fail_counts.items() if a != most_common),
        ],
    ]

    # Handle edge case where a row or column is all zeros
    if any(sum(row) == 0 for row in table) or any(
        table[0][i] + table[1][i] == 0 for i in range(2)
    ):
        p_value = 1.0
    else:
        _, p_value = stats.fisher_exact(table)

    return {
        "step_index": step_index,
        "p_value": float(p_value),
        "successful_actions": dict(success_counts),
        "failed_actions": dict(fail_counts),
        "most_common_success": success_counts.most_common(1)[0][0] if success_counts else None,
        "most_common_fail": fail_counts.most_common(1)[0][0] if fail_counts else None,
    }


def attribute_failures(trials: list[TrialResult]) -> dict[str, Any]:
    """Attribute failures to specific trajectory steps.

    Analyzes all steps to find which step shows the most significant
    divergence between successful and failed trials.

    Args:
        trials: All trial results (mix of passed and failed).

    Returns:
        Dictionary with:
        - most_likely_step: Step index most likely causing failures
        - step_divergences: Divergence info for each step
        - failure_patterns: Common failure patterns
        - recommendation: Human-readable recommendation
    """
    successful = [t for t in trials if t.passed]
    failed = [t for t in trials if not t.passed]

    if not failed:
        return {
            "most_likely_step": None,
            "step_divergences": [],
            "failure_patterns": [],
            "recommendation": "No failures to analyze",
        }

    if not successful:
        return {
            "most_likely_step": None,
            "step_divergences": [],
            "failure_patterns": _extract_failure_patterns(failed),
            "recommendation": "All trials failed. Check agent configuration and expectations.",
        }

    # Find max trajectory length
    max_steps = max(len(t.agent_output.steps) for t in trials)

    # Compute divergence for each step
    step_divergences = []
    for i in range(max_steps):
        div = compute_divergence(successful, failed, i)
        step_divergences.append(div)

    # Apply Benjamini-Hochberg correction for multiple testing
    raw_p_values = [d["p_value"] for d in step_divergences]
    significant_flags, adjusted_p_values = benjamini_hochberg_correction(raw_p_values, alpha=0.05)

    for i, div in enumerate(step_divergences):
        div["raw_p_value"] = div["p_value"]
        div["p_value"] = adjusted_p_values[i]
        div["significant"] = significant_flags[i]

    # Find step with lowest adjusted p-value (most significant divergence)
    significant_steps = [d for d in step_divergences if d["significant"]]

    if not significant_steps:
        most_likely_step = None
        recommendation = (
            "No single step shows significant divergence after "
            "Benjamini-Hochberg correction. "
            "Failures may be due to cumulative errors or random variation."
        )
    else:
        most_likely = min(step_divergences, key=lambda d: d["p_value"])
        most_likely_step = most_likely["step_index"]

        # Build recommendation
        if most_likely["p_value"] < 0.05:
            significance = "highly significant"
        else:
            significance = "moderately significant"

        success_action = most_likely.get("most_common_success", "unknown")
        fail_action = most_likely.get("most_common_fail", "unknown")

        recommendation = (
            f"Step {most_likely_step} shows {significance} divergence "
            f"(adjusted p={most_likely['p_value']:.3f}). "
            f"Successful runs typically use '{success_action}', "
            f"while failed runs use '{fail_action}'."
        )

    return {
        "most_likely_step": most_likely_step,
        "step_divergences": step_divergences,
        "failure_patterns": _extract_failure_patterns(failed),
        "recommendation": recommendation,
    }


def _extract_failure_patterns(failed_trials: list[TrialResult]) -> list[dict[str, Any]]:
    """Extract common failure patterns from failed trials.

    Args:
        failed_trials: List of failed trial results.

    Returns:
        List of failure patterns with counts.
    """
    # Group by failure message
    failure_counts: Counter[str] = Counter()
    for trial in failed_trials:
        for failure in trial.failures:
            failure_counts[failure] += 1

    patterns = []
    for message, count in failure_counts.most_common(5):
        patterns.append(
            {
                "message": message,
                "count": count,
                "percentage": count / len(failed_trials) * 100,
            }
        )

    return patterns


def compare_trajectories(
    trajectory_a: list[dict[str, Any]],
    trajectory_b: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare two trajectories step by step.

    Args:
        trajectory_a: First trajectory (list of steps).
        trajectory_b: Second trajectory (list of steps).

    Returns:
        Dictionary with:
        - length_diff: Difference in trajectory length
        - first_divergence: Index of first diverging step
        - divergent_steps: List of indices where steps differ
        - similarity_score: Overall similarity (0-1)
    """
    len_a = len(trajectory_a)
    len_b = len(trajectory_b)

    # Find divergent steps
    divergent_steps = []
    first_divergence = None
    matches = 0

    for i in range(min(len_a, len_b)):
        step_a = trajectory_a[i]
        step_b = trajectory_b[i]

        # Compare step type and name
        if step_a.get("name") == step_b.get("name") and step_a.get("step_type") == step_b.get(
            "step_type"
        ):
            matches += 1
        else:
            divergent_steps.append(i)
            if first_divergence is None:
                first_divergence = i

    # Account for length difference
    divergent_steps.extend(range(min(len_a, len_b), max(len_a, len_b)))

    max_len = max(len_a, len_b)
    similarity = matches / max_len if max_len > 0 else 1.0

    return {
        "length_diff": len_a - len_b,
        "first_divergence": first_divergence,
        "divergent_steps": divergent_steps,
        "similarity_score": similarity,
    }
