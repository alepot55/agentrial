"""Agent Reliability Score (ARS).

Computes a single 0-100 composite score from multiple evaluation
dimensions. Useful for comparing agents across different suites and
over time.

ARS = weighted sum of 6 components:
  - Accuracy (40%): overall pass rate
  - Consistency (20%): 1 - coefficient of variation of per-case pass rates
  - Cost efficiency (10%): normalized against a budget ceiling
  - Latency (10%): normalized against a latency ceiling
  - Trajectory quality (10%): avg step success rate
  - Recovery (10%): fraction of cases that recovered (passed after a failure)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentrial.types import SuiteResult


@dataclass
class ARSBreakdown:
    """Detailed breakdown of the ARS score."""

    score: float
    accuracy: float
    consistency: float
    cost_efficiency: float
    latency: float
    trajectory_quality: float
    recovery: float
    weights: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


DEFAULT_WEIGHTS = {
    "accuracy": 0.40,
    "consistency": 0.20,
    "cost_efficiency": 0.10,
    "latency": 0.10,
    "trajectory_quality": 0.10,
    "recovery": 0.10,
}


def compute_ars(
    result: SuiteResult,
    *,
    cost_ceiling: float = 1.0,
    latency_ceiling_ms: float = 30000.0,
    weights: dict[str, float] | None = None,
) -> ARSBreakdown:
    """Compute Agent Reliability Score from a SuiteResult.

    Args:
        result: The evaluated suite result.
        cost_ceiling: Maximum expected cost per case (for normalization).
        latency_ceiling_ms: Maximum expected latency in ms (for normalization).
        weights: Custom component weights (must sum to 1.0).

    Returns:
        ARSBreakdown with the overall score and per-component values.
    """
    w = weights or DEFAULT_WEIGHTS

    # 1. Accuracy: overall pass rate (0 to 1)
    accuracy = result.overall_pass_rate

    # 2. Consistency: 1 - CV of per-case pass rates
    case_pass_rates = [r.pass_rate for r in result.results]
    if len(case_pass_rates) > 1:
        mean_pr = sum(case_pass_rates) / len(case_pass_rates)
        if mean_pr > 0:
            variance = sum((p - mean_pr) ** 2 for p in case_pass_rates) / len(case_pass_rates)
            cv = variance**0.5 / mean_pr
            consistency = max(0.0, 1.0 - cv)
        else:
            consistency = 0.0
    elif len(case_pass_rates) == 1:
        consistency = 1.0
    else:
        consistency = 0.0

    # 3. Cost efficiency: 1 - (avg_cost / ceiling), clamped to [0, 1]
    num_cases = max(len(result.results), 1)
    avg_cost = result.total_cost / num_cases
    cost_efficiency = max(0.0, min(1.0, 1.0 - avg_cost / cost_ceiling))

    # 4. Latency: 1 - (avg_latency / ceiling), clamped to [0, 1]
    avg_latency = result.total_duration_ms / num_cases
    latency_score = max(0.0, min(1.0, 1.0 - avg_latency / latency_ceiling_ms))

    # 5. Trajectory quality: fraction of steps that produced valid output
    total_steps = 0
    successful_steps = 0
    for eval_result in result.results:
        for trial in eval_result.trials:
            for step in trial.agent_output.steps:
                total_steps += 1
                if step.output is not None:
                    successful_steps += 1
    trajectory_quality = successful_steps / total_steps if total_steps > 0 else 1.0

    # 6. Recovery: fraction of cases where at least one trial passed
    #    after another trial failed (indicates ability to self-recover)
    recovery_count = 0
    for eval_result in result.results:
        has_fail = any(not t.passed for t in eval_result.trials)
        has_pass = any(t.passed for t in eval_result.trials)
        if has_fail and has_pass:
            recovery_count += 1
    recovery = recovery_count / num_cases if num_cases > 0 else 0.0

    # Weighted sum
    score = (
        w.get("accuracy", 0.4) * accuracy
        + w.get("consistency", 0.2) * consistency
        + w.get("cost_efficiency", 0.1) * cost_efficiency
        + w.get("latency", 0.1) * latency_score
        + w.get("trajectory_quality", 0.1) * trajectory_quality
        + w.get("recovery", 0.1) * recovery
    ) * 100

    return ARSBreakdown(
        score=round(score, 1),
        accuracy=round(accuracy * 100, 1),
        consistency=round(consistency * 100, 1),
        cost_efficiency=round(cost_efficiency * 100, 1),
        latency=round(latency_score * 100, 1),
        trajectory_quality=round(trajectory_quality * 100, 1),
        recovery=round(recovery * 100, 1),
        weights=dict(w),
        details={
            "num_cases": len(result.results),
            "avg_cost": round(avg_cost, 6),
            "avg_latency_ms": round(avg_latency, 1),
            "total_steps": total_steps,
            "cost_ceiling": cost_ceiling,
            "latency_ceiling_ms": latency_ceiling_ms,
        },
    )


def compute_ars_from_json(
    report: dict[str, Any],
    *,
    cost_ceiling: float = 1.0,
    latency_ceiling_ms: float = 30000.0,
) -> ARSBreakdown:
    """Compute ARS from a JSON report (exported by export_json).

    This allows computing ARS from saved results without re-running.
    """
    summary = report.get("summary", {})
    results = report.get("results", [])

    accuracy = summary.get("overall_pass_rate", 0.0)

    # Consistency
    case_pass_rates = [r.get("pass_rate", 0.0) for r in results]
    if len(case_pass_rates) > 1:
        mean_pr = sum(case_pass_rates) / len(case_pass_rates)
        if mean_pr > 0:
            variance = sum((p - mean_pr) ** 2 for p in case_pass_rates) / len(case_pass_rates)
            cv = variance**0.5 / mean_pr
            consistency = max(0.0, 1.0 - cv)
        else:
            consistency = 0.0
    elif len(case_pass_rates) == 1:
        consistency = 1.0
    else:
        consistency = 0.0

    # Cost
    total_cost = summary.get("total_cost", 0.0)
    num_cases = max(len(results), 1)
    avg_cost = total_cost / num_cases
    cost_efficiency = max(0.0, min(1.0, 1.0 - avg_cost / cost_ceiling))

    # Latency
    total_duration = summary.get("total_duration_ms", 0.0)
    avg_latency = total_duration / num_cases
    latency_score = max(0.0, min(1.0, 1.0 - avg_latency / latency_ceiling_ms))

    # Trajectory quality and recovery from JSON are approximated
    trajectory_quality = 1.0
    recovery = 0.0
    for r in results:
        pr = r.get("pass_rate", 0.0)
        if 0 < pr < 1.0:
            recovery += 1
    recovery = recovery / num_cases if num_cases > 0 else 0.0

    score = (
        0.40 * accuracy
        + 0.20 * consistency
        + 0.10 * cost_efficiency
        + 0.10 * latency_score
        + 0.10 * trajectory_quality
        + 0.10 * recovery
    ) * 100

    return ARSBreakdown(
        score=round(score, 1),
        accuracy=round(accuracy * 100, 1),
        consistency=round(consistency * 100, 1),
        cost_efficiency=round(cost_efficiency * 100, 1),
        latency=round(latency_score * 100, 1),
        trajectory_quality=round(trajectory_quality * 100, 1),
        recovery=round(recovery * 100, 1),
        weights=dict(DEFAULT_WEIGHTS),
        details={
            "num_cases": len(results),
            "avg_cost": round(avg_cost, 6),
            "avg_latency_ms": round(avg_latency, 1),
            "cost_ceiling": cost_ceiling,
            "latency_ceiling_ms": latency_ceiling_ms,
            "source": "json",
        },
    )
