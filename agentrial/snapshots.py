"""Snapshot testing for agents.

Captures statistical distributions (pass rate, cost, latency, step actions)
as a snapshot baseline and compares subsequent runs against it using the
statistical tests already implemented (Fisher, Mann-Whitney U).

Inspired by Jest snapshot testing, adapted for non-deterministic agents.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentrial.metrics.statistical import (
    fisher_exact_test,
    mann_whitney_u_test,
)
from agentrial.types import EvalResult, SuiteResult

SNAPSHOT_DIR = ".agentrial/snapshots"
SNAPSHOT_VERSION = "1.0"


def create_snapshot(suite_result: SuiteResult) -> dict[str, Any]:
    """Create a snapshot from suite results.

    The snapshot captures the statistical distribution of metrics
    so future runs can be compared for regressions.

    Args:
        suite_result: Results from running a test suite.

    Returns:
        JSON-serializable snapshot dict.
    """
    cases = []
    for result in suite_result.results:
        case_snap = _snapshot_eval_result(result)
        cases.append(case_snap)

    return {
        "version": SNAPSHOT_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "suite": suite_result.suite.name,
        "trials": suite_result.suite.trials,
        "threshold": suite_result.suite.threshold,
        "overall": {
            "pass_rate": suite_result.overall_pass_rate,
            "pass_rate_ci": {
                "lower": suite_result.overall_pass_rate_ci.lower,
                "upper": suite_result.overall_pass_rate_ci.upper,
            },
            "total_cost": suite_result.total_cost,
            "total_duration_ms": suite_result.total_duration_ms,
        },
        "cases": cases,
    }


def _snapshot_eval_result(result: EvalResult) -> dict[str, Any]:
    """Create a snapshot for a single eval result."""
    trials = result.trials
    costs = [t.cost for t in trials]
    latencies = [t.duration_ms for t in trials]

    # Step action distributions
    step_distributions: dict[str, dict[str, Any]] = {}
    max_steps = max((len(t.agent_output.steps) for t in trials), default=0)

    for step_idx in range(max_steps):
        action_counts: dict[str, int] = {}
        present = 0
        for trial in trials:
            if step_idx < len(trial.agent_output.steps):
                step = trial.agent_output.steps[step_idx]
                action_counts[step.name] = action_counts.get(step.name, 0) + 1
                present += 1

        if present > 0:
            # Normalize to rates
            action_rates = {
                name: count / present for name, count in action_counts.items()
            }
            step_distributions[f"step_{step_idx}"] = {
                "presence_rate": present / len(trials),
                "top_actions": action_rates,
            }

    return {
        "name": result.test_case.name,
        "trials": len(trials),
        "metrics": {
            "pass_rate": {
                "mean": result.pass_rate,
                "ci_lower": result.pass_rate_ci.lower,
                "ci_upper": result.pass_rate_ci.upper,
            },
            "cost": {
                "values": costs,
                "mean": result.mean_cost,
                "ci_lower": result.cost_ci.lower,
                "ci_upper": result.cost_ci.upper,
            },
            "latency": {
                "values": latencies,
                "mean": result.mean_latency_ms,
                "ci_lower": result.latency_ci.lower,
                "ci_upper": result.latency_ci.upper,
            },
        },
        "step_distributions": step_distributions,
    }


def save_snapshot(
    snapshot: dict[str, Any],
    path: Path | str | None = None,
    suite_name: str | None = None,
) -> Path:
    """Save a snapshot to disk.

    Args:
        snapshot: Snapshot data to save.
        path: Explicit path. If None, saves to .agentrial/snapshots/<suite>.json.
        suite_name: Suite name for default path. Uses snapshot data if not given.

    Returns:
        Path where snapshot was saved.
    """
    if path is None:
        name = suite_name or snapshot.get("suite", "default")
        path = Path(SNAPSHOT_DIR) / f"{name}.json"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)

    return path


def load_snapshot(path: Path | str) -> dict[str, Any]:
    """Load a snapshot from disk.

    Args:
        path: Path to snapshot file.

    Returns:
        Snapshot data.

    Raises:
        FileNotFoundError: If snapshot file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot not found: {path}")

    with open(path) as f:
        return json.load(f)


def find_snapshot(suite_name: str) -> Path | None:
    """Find the snapshot file for a suite.

    Args:
        suite_name: Suite name to find snapshot for.

    Returns:
        Path to snapshot file, or None if not found.
    """
    path = Path(SNAPSHOT_DIR) / f"{suite_name}.json"
    return path if path.exists() else None


def compare_with_snapshot(
    current: SuiteResult,
    snapshot: dict[str, Any],
    alpha: float = 0.05,
) -> SnapshotComparison:
    """Compare current results against a snapshot baseline.

    Uses Fisher exact test for pass rate and Mann-Whitney U for
    cost/latency distributions.

    Args:
        current: Current suite results.
        snapshot: Baseline snapshot data.
        alpha: Significance level for tests.

    Returns:
        SnapshotComparison with per-case results.
    """
    baseline_cases = {c["name"]: c for c in snapshot.get("cases", [])}
    case_comparisons = []

    for result in current.results:
        baseline = baseline_cases.get(result.test_case.name)
        if baseline is None:
            case_comparisons.append(
                CaseComparison(
                    name=result.test_case.name,
                    status="new",
                    current_pass_rate=result.pass_rate,
                )
            )
            continue

        comp = _compare_case(result, baseline, alpha)
        case_comparisons.append(comp)

    # Check for removed cases
    current_names = {r.test_case.name for r in current.results}
    for name in baseline_cases:
        if name not in current_names:
            case_comparisons.append(
                CaseComparison(
                    name=name,
                    status="removed",
                    baseline_pass_rate=baseline_cases[name]["metrics"]["pass_rate"]["mean"],
                )
            )

    # Overall status
    has_regression = any(c.status == "regression" for c in case_comparisons)

    return SnapshotComparison(
        suite_name=snapshot.get("suite", "unknown"),
        baseline_date=snapshot.get("created_at", ""),
        cases=case_comparisons,
        overall_passed=not has_regression,
    )


def _compare_case(
    current: EvalResult,
    baseline: dict[str, Any],
    alpha: float,
) -> CaseComparison:
    """Compare a single test case against its snapshot."""
    metrics = baseline.get("metrics", {})
    baseline_pass_rate = metrics.get("pass_rate", {}).get("mean", 0.0)
    baseline_trials = baseline.get("trials", 10)

    # Fisher exact test for pass rate
    current_passed = sum(1 for t in current.trials if t.passed)
    current_total = len(current.trials)
    baseline_passed = round(baseline_pass_rate * baseline_trials)

    _, pass_rate_p = fisher_exact_test(
        current_passed, current_total, baseline_passed, baseline_trials
    )

    pass_rate_delta = current.pass_rate - baseline_pass_rate

    # Cost comparison (Mann-Whitney U if we have raw values)
    cost_delta = None
    cost_p = None
    baseline_costs = metrics.get("cost", {}).get("values")
    if baseline_costs:
        current_costs = [t.cost for t in current.trials]
        _, cost_p = mann_whitney_u_test(current_costs, baseline_costs)
        cost_delta = current.mean_cost - metrics["cost"]["mean"]

    # Latency comparison
    latency_delta = None
    latency_p = None
    baseline_latencies = metrics.get("latency", {}).get("values")
    if baseline_latencies:
        current_latencies = [t.duration_ms for t in current.trials]
        _, latency_p = mann_whitney_u_test(current_latencies, baseline_latencies)
        latency_delta = current.mean_latency_ms - metrics["latency"]["mean"]

    # Determine status
    is_pass_rate_regression = pass_rate_delta < 0 and pass_rate_p < alpha
    is_cost_regression = (
        cost_delta is not None
        and cost_p is not None
        and cost_delta > 0
        and cost_p < alpha
    )
    is_latency_regression = (
        latency_delta is not None
        and latency_p is not None
        and latency_delta > 0
        and latency_p < alpha
    )

    if is_pass_rate_regression:
        status = "regression"
    elif cost_delta is not None and is_cost_regression:
        status = "cost_regression"
    elif latency_delta is not None and is_latency_regression:
        status = "latency_regression"
    elif pass_rate_delta > 0 and pass_rate_p < alpha:
        status = "improved"
    else:
        status = "no_change"

    return CaseComparison(
        name=current.test_case.name,
        status=status,
        current_pass_rate=current.pass_rate,
        baseline_pass_rate=baseline_pass_rate,
        pass_rate_delta=pass_rate_delta,
        pass_rate_p=pass_rate_p,
        cost_delta=cost_delta,
        cost_p=cost_p,
        latency_delta=latency_delta,
        latency_p=latency_p,
    )


class CaseComparison:
    """Result of comparing a test case against its snapshot."""

    def __init__(
        self,
        name: str,
        status: str,
        current_pass_rate: float | None = None,
        baseline_pass_rate: float | None = None,
        pass_rate_delta: float | None = None,
        pass_rate_p: float | None = None,
        cost_delta: float | None = None,
        cost_p: float | None = None,
        latency_delta: float | None = None,
        latency_p: float | None = None,
    ) -> None:
        self.name = name
        self.status = status
        self.current_pass_rate = current_pass_rate
        self.baseline_pass_rate = baseline_pass_rate
        self.pass_rate_delta = pass_rate_delta
        self.pass_rate_p = pass_rate_p
        self.cost_delta = cost_delta
        self.cost_p = cost_p
        self.latency_delta = latency_delta
        self.latency_p = latency_p

    @property
    def is_regression(self) -> bool:
        return self.status in ("regression", "cost_regression", "latency_regression")


class SnapshotComparison:
    """Result of comparing a full suite against a snapshot."""

    def __init__(
        self,
        suite_name: str,
        baseline_date: str,
        cases: list[CaseComparison],
        overall_passed: bool,
    ) -> None:
        self.suite_name = suite_name
        self.baseline_date = baseline_date
        self.cases = cases
        self.overall_passed = overall_passed

    @property
    def regressions(self) -> list[CaseComparison]:
        return [c for c in self.cases if c.is_regression]

    @property
    def improvements(self) -> list[CaseComparison]:
        return [c for c in self.cases if c.status == "improved"]


def print_snapshot_comparison(
    comparison: SnapshotComparison,
    console: Any = None,
) -> None:
    """Print snapshot comparison results to terminal.

    Args:
        comparison: Comparison results.
        console: Rich console (creates one if not provided).
    """
    from rich.console import Console
    from rich.table import Table

    if console is None:
        console = Console()

    status = (
        "[green]PASSED[/green]"
        if comparison.overall_passed
        else "[red]REGRESSION DETECTED[/red]"
    )
    console.print(
        f"\n[bold]Snapshot comparison:[/bold] {comparison.suite_name} — {status}"
    )
    if comparison.baseline_date:
        console.print(f"[dim]Baseline: {comparison.baseline_date}[/dim]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Test Case", min_width=25, no_wrap=True)
    table.add_column("Status", min_width=12)
    table.add_column("Current", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("p-value", justify="right")

    for case in comparison.cases:
        status_text = {
            "regression": "[red]REGRESSION[/red]",
            "cost_regression": "[red]COST UP[/red]",
            "latency_regression": "[red]SLOWER[/red]",
            "improved": "[green]IMPROVED[/green]",
            "no_change": "[dim]no change[/dim]",
            "new": "[blue]NEW[/blue]",
            "removed": "[yellow]REMOVED[/yellow]",
        }.get(case.status, case.status)

        current = f"{case.current_pass_rate:.0%}" if case.current_pass_rate is not None else "-"
        baseline = (
            f"{case.baseline_pass_rate:.0%}" if case.baseline_pass_rate is not None else "-"
        )

        if case.pass_rate_delta is not None:
            delta = f"{case.pass_rate_delta:+.0%}"
            delta_style = "red" if case.pass_rate_delta < 0 else "green"
            delta_text = f"[{delta_style}]{delta}[/{delta_style}]"
        else:
            delta_text = "-"

        p_val = f"{case.pass_rate_p:.3f}" if case.pass_rate_p is not None else "-"

        table.add_row(case.name, status_text, current, baseline, delta_text, p_val)

    console.print(table)
