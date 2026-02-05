"""Basic metrics computation for AgentEval."""

from typing import Any

from agenteval.types import TrialResult


def compute_basic_metrics(trials: list[TrialResult]) -> dict[str, Any]:
    """Compute basic metrics from trial results.

    Args:
        trials: List of trial results.

    Returns:
        Dictionary containing:
        - pass_count: Number of passing trials
        - fail_count: Number of failing trials
        - pass_rate: Fraction of passing trials
        - mean_cost: Mean cost across trials
        - total_cost: Total cost across all trials
        - mean_latency_ms: Mean latency in milliseconds
        - mean_tokens: Mean token count
        - total_tokens: Total tokens across all trials
    """
    if not trials:
        return {
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": 0.0,
            "mean_cost": 0.0,
            "total_cost": 0.0,
            "mean_latency_ms": 0.0,
            "mean_tokens": 0.0,
            "total_tokens": 0,
        }

    pass_count = sum(1 for t in trials if t.passed)
    fail_count = len(trials) - pass_count

    costs = [t.cost for t in trials]
    latencies = [t.duration_ms for t in trials]
    tokens = [t.tokens for t in trials]

    return {
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": pass_count / len(trials),
        "mean_cost": sum(costs) / len(costs) if costs else 0.0,
        "total_cost": sum(costs),
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "mean_tokens": sum(tokens) / len(tokens) if tokens else 0.0,
        "total_tokens": sum(tokens),
    }


def compute_cost_per_correct(trials: list[TrialResult]) -> float:
    """Compute cost per correct answer.

    This is a key efficiency metric: how much does it cost on average
    to get a correct answer from the agent?

    Args:
        trials: List of trial results.

    Returns:
        Cost per correct answer, or infinity if no correct answers.
    """
    passed_trials = [t for t in trials if t.passed]
    if not passed_trials:
        return float("inf")

    total_cost = sum(t.cost for t in trials)
    return total_cost / len(passed_trials)


def compute_latency_percentiles(
    trials: list[TrialResult],
    percentiles: list[float] | None = None,
) -> dict[str, float]:
    """Compute latency percentiles.

    Args:
        trials: List of trial results.
        percentiles: Percentiles to compute (default: [50, 90, 95, 99]).

    Returns:
        Dictionary mapping percentile names to values.
    """
    import numpy as np

    if percentiles is None:
        percentiles = [50.0, 90.0, 95.0, 99.0]

    latencies = [t.duration_ms for t in trials]
    if not latencies:
        return {f"p{int(p)}": 0.0 for p in percentiles}

    result = {}
    for p in percentiles:
        result[f"p{int(p)}"] = float(np.percentile(latencies, p))

    return result


def compute_token_efficiency(trials: list[TrialResult]) -> dict[str, float]:
    """Compute token efficiency metrics.

    Args:
        trials: List of trial results.

    Returns:
        Dictionary containing:
        - tokens_per_trial: Average tokens per trial
        - tokens_per_correct: Average tokens per correct answer
        - tokens_per_step: Average tokens per trajectory step
    """
    if not trials:
        return {
            "tokens_per_trial": 0.0,
            "tokens_per_correct": float("inf"),
            "tokens_per_step": 0.0,
        }

    total_tokens = sum(t.tokens for t in trials)
    total_steps = sum(len(t.agent_output.steps) for t in trials)
    passed_count = sum(1 for t in trials if t.passed)

    return {
        "tokens_per_trial": total_tokens / len(trials),
        "tokens_per_correct": total_tokens / passed_count if passed_count > 0 else float("inf"),
        "tokens_per_step": total_tokens / total_steps if total_steps > 0 else 0.0,
    }
