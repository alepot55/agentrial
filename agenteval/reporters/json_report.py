"""JSON export for AgentEval results."""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from agenteval.types import EvalResult, SuiteResult


def _serialize_value(obj: Any) -> Any:
    """Serialize a value for JSON export."""
    if hasattr(obj, "__dict__"):
        # Dataclass or similar
        return {k: _serialize_value(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    elif isinstance(obj, list):
        return [_serialize_value(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _serialize_value(v) for k, v in obj.items()}
    elif hasattr(obj, "value"):
        # Enum
        return obj.value
    elif callable(obj):
        # Skip callables (custom check functions)
        return None
    return obj


def export_json(suite_result: SuiteResult) -> dict[str, Any]:
    """Export suite result as JSON-serializable dict.

    Args:
        suite_result: The suite result to export.

    Returns:
        JSON-serializable dictionary.
    """
    return {
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "suite": {
            "name": suite_result.suite.name,
            "agent": suite_result.suite.agent,
            "trials": suite_result.suite.trials,
            "threshold": suite_result.suite.threshold,
        },
        "summary": {
            "passed": suite_result.passed,
            "overall_pass_rate": suite_result.overall_pass_rate,
            "overall_pass_rate_ci": {
                "lower": suite_result.overall_pass_rate_ci.lower,
                "upper": suite_result.overall_pass_rate_ci.upper,
                "confidence_level": suite_result.overall_pass_rate_ci.confidence_level,
            },
            "total_cost": suite_result.total_cost,
            "total_duration_ms": suite_result.total_duration_ms,
        },
        "results": [_export_eval_result(r) for r in suite_result.results],
    }


def _export_eval_result(result: EvalResult) -> dict[str, Any]:
    """Export a single evaluation result."""
    return {
        "test_case": {
            "name": result.test_case.name,
            "tags": result.test_case.tags,
        },
        "pass_rate": result.pass_rate,
        "pass_rate_ci": {
            "lower": result.pass_rate_ci.lower,
            "upper": result.pass_rate_ci.upper,
        },
        "mean_cost": result.mean_cost,
        "cost_ci": {
            "lower": result.cost_ci.lower,
            "upper": result.cost_ci.upper,
        },
        "mean_latency_ms": result.mean_latency_ms,
        "latency_ci": {
            "lower": result.latency_ci.lower,
            "upper": result.latency_ci.upper,
        },
        "mean_tokens": result.mean_tokens,
        "trials": [_export_trial(t) for t in result.trials],
        "failure_attribution": _serialize_value(result.failure_attribution),
    }


def _export_trial(trial: Any) -> dict[str, Any]:
    """Export a single trial result."""
    return {
        "trial_index": trial.trial_index,
        "passed": trial.passed,
        "failures": trial.failures,
        "duration_ms": trial.duration_ms,
        "cost": trial.cost,
        "tokens": trial.tokens,
        "steps": [_export_step(s) for s in trial.agent_output.steps],
    }


def _export_step(step: Any) -> dict[str, Any]:
    """Export a trajectory step."""
    return {
        "step_index": step.step_index,
        "step_type": step.step_type.value,
        "name": step.name,
        "duration_ms": step.duration_ms,
        "tokens": step.tokens,
    }


def save_json_report(suite_result: SuiteResult, path: Path | str) -> None:
    """Save suite result to a JSON file.

    Args:
        suite_result: The suite result to save.
        path: Output file path.
    """
    path = Path(path)
    data = export_json(suite_result)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json_report(path: Path | str) -> dict[str, Any]:
    """Load a JSON report from file.

    Args:
        path: Path to the JSON report.

    Returns:
        Loaded report data.
    """
    path = Path(path)
    with open(path) as f:
        return json.load(f)


def compare_reports(
    current: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Compare two JSON reports.

    Args:
        current: Current run report.
        baseline: Baseline report.

    Returns:
        Comparison summary.
    """
    current_results = {r["test_case"]["name"]: r for r in current["results"]}
    baseline_results = {r["test_case"]["name"]: r for r in baseline["results"]}

    comparisons = []

    for name, current_result in current_results.items():
        if name in baseline_results:
            baseline_result = baseline_results[name]
            delta = current_result["pass_rate"] - baseline_result["pass_rate"]
            comparisons.append(
                {
                    "test_case": name,
                    "current_pass_rate": current_result["pass_rate"],
                    "baseline_pass_rate": baseline_result["pass_rate"],
                    "delta": delta,
                    "status": "regression" if delta < -0.1 else ("improved" if delta > 0.1 else "stable"),
                }
            )
        else:
            comparisons.append(
                {
                    "test_case": name,
                    "current_pass_rate": current_result["pass_rate"],
                    "baseline_pass_rate": None,
                    "delta": None,
                    "status": "new",
                }
            )

    # Check for removed tests
    for name in baseline_results:
        if name not in current_results:
            comparisons.append(
                {
                    "test_case": name,
                    "current_pass_rate": None,
                    "baseline_pass_rate": baseline_results[name]["pass_rate"],
                    "delta": None,
                    "status": "removed",
                }
            )

    return {
        "overall_delta": current["summary"]["overall_pass_rate"] - baseline["summary"]["overall_pass_rate"],
        "current_passed": current["summary"]["passed"],
        "baseline_passed": baseline["summary"]["passed"],
        "comparisons": comparisons,
    }
