"""Cost-Accuracy Pareto Frontier analysis.

Runs a test suite against multiple models and identifies which
models are Pareto-optimal (not dominated in both cost AND accuracy).

Usage:
    from agentrial.pareto import ParetoAnalyzer
    analyzer = ParetoAnalyzer(suite, agent_factory, models)
    result = analyzer.analyze(trials=20)
    print_pareto(result)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelPoint:
    """A single model's evaluation on the Pareto frontier."""

    model_name: str
    pass_rate: float
    mean_cost: float
    mean_latency_ms: float
    ci_lower: float = 0.0
    ci_upper: float = 1.0
    trials: int = 0
    dominated: bool = False
    pareto_optimal: bool = False


@dataclass
class ParetoResult:
    """Complete Pareto analysis result."""

    points: list[ModelPoint] = field(default_factory=list)
    pareto_frontier: list[ModelPoint] = field(default_factory=list)
    dominated: list[ModelPoint] = field(default_factory=list)
    recommendation: str = ""

    @property
    def best_quality(self) -> ModelPoint | None:
        """Pareto-optimal model with highest pass rate."""
        if not self.pareto_frontier:
            return None
        return max(self.pareto_frontier, key=lambda p: p.pass_rate)

    @property
    def best_value(self) -> ModelPoint | None:
        """Pareto-optimal model with lowest cost."""
        if not self.pareto_frontier:
            return None
        return min(self.pareto_frontier, key=lambda p: p.mean_cost)


def compute_pareto_frontier(points: list[ModelPoint]) -> list[ModelPoint]:
    """Identify the Pareto-optimal models.

    A model is Pareto-optimal if no other model has both:
    - Higher or equal pass rate AND lower or equal cost

    Args:
        points: List of model evaluation points.

    Returns:
        List of Pareto-optimal points (also marks dominated/optimal flags).
    """
    if not points:
        return []

    frontier = []

    for p in points:
        is_dominated = False
        for q in points:
            if p is q:
                continue
            # q dominates p if q is at least as good in both dimensions
            # and strictly better in at least one
            if (
                q.pass_rate >= p.pass_rate
                and q.mean_cost <= p.mean_cost
                and (q.pass_rate > p.pass_rate or q.mean_cost < p.mean_cost)
            ):
                is_dominated = True
                break

        p.dominated = is_dominated
        p.pareto_optimal = not is_dominated

        if not is_dominated:
            frontier.append(p)

    # Sort frontier by cost (ascending)
    frontier.sort(key=lambda p: p.mean_cost)
    return frontier


def generate_recommendation(result: ParetoResult) -> str:
    """Generate a human-readable recommendation from Pareto analysis.

    Args:
        result: The Pareto analysis result.

    Returns:
        Recommendation string.
    """
    if not result.pareto_frontier:
        return "No models evaluated."

    if len(result.pareto_frontier) == 1:
        p = result.pareto_frontier[0]
        return (
            f"{p.model_name} is the only Pareto-optimal model "
            f"({p.pass_rate:.0%} pass rate at ${p.mean_cost:.4f}/run)."
        )

    best_q = result.best_quality
    best_v = result.best_value

    if best_q is None or best_v is None:
        return "Unable to determine recommendation."

    parts = []
    parts.append(
        f"For quality: {best_q.model_name} "
        f"({best_q.pass_rate:.0%} at ${best_q.mean_cost:.4f}/run)"
    )
    parts.append(
        f"For cost: {best_v.model_name} "
        f"({best_v.pass_rate:.0%} at ${best_v.mean_cost:.4f}/run)"
    )

    # Find dominated models and suggest switches
    for d in result.dominated:
        # Find the frontier point that dominates it
        for f in result.pareto_frontier:
            if f.pass_rate >= d.pass_rate and f.mean_cost <= d.mean_cost:
                if d.mean_cost > 0:
                    savings = (1 - f.mean_cost / d.mean_cost) * 100
                    delta = f.pass_rate - d.pass_rate
                    parts.append(
                        f"Switch from {d.model_name} to {f.model_name}: "
                        f"save {savings:.0f}% cost"
                        f"{f' with +{delta:.0%} accuracy' if delta > 0 else ''}"
                    )
                break

    return " | ".join(parts)


def analyze_from_results(
    model_results: dict[str, dict[str, Any]],
) -> ParetoResult:
    """Analyze Pareto frontier from pre-computed model results.

    Args:
        model_results: Dict mapping model_name to metrics dict with keys:
            - pass_rate (float)
            - mean_cost (float)
            - mean_latency_ms (float)
            - ci_lower (float, optional)
            - ci_upper (float, optional)
            - trials (int, optional)

    Returns:
        ParetoResult with frontier analysis.
    """
    points = []
    for model_name, metrics in model_results.items():
        points.append(
            ModelPoint(
                model_name=model_name,
                pass_rate=metrics["pass_rate"],
                mean_cost=metrics["mean_cost"],
                mean_latency_ms=metrics.get("mean_latency_ms", 0.0),
                ci_lower=metrics.get("ci_lower", 0.0),
                ci_upper=metrics.get("ci_upper", 1.0),
                trials=metrics.get("trials", 0),
            )
        )

    frontier = compute_pareto_frontier(points)
    dominated = [p for p in points if p.dominated]

    result = ParetoResult(
        points=points,
        pareto_frontier=frontier,
        dominated=dominated,
    )

    result.recommendation = generate_recommendation(result)
    return result


def render_pareto_ascii(result: ParetoResult, width: int = 60) -> str:
    """Render the Pareto frontier as an ASCII scatter plot.

    Args:
        result: Pareto analysis result.
        width: Width of the plot in characters.

    Returns:
        Multi-line ASCII string.
    """
    if not result.points:
        return "No data to plot."

    # Determine axis ranges
    min_cost = min(p.mean_cost for p in result.points)
    max_cost = max(p.mean_cost for p in result.points)
    min_rate = min(p.pass_rate for p in result.points)
    max_rate = max(p.pass_rate for p in result.points)

    # Ensure non-zero ranges
    if max_cost == min_cost:
        max_cost = min_cost + 0.01
    if max_rate == min_rate:
        max_rate = min_rate + 0.1

    height = 15
    lines = []

    # Header
    lines.append("Cost-Accuracy Pareto Frontier")
    lines.append("=" * width)

    # Create grid
    grid: list[list[str]] = [[" "] * width for _ in range(height)]

    # Plot points
    for p in result.points:
        x = int((p.mean_cost - min_cost) / (max_cost - min_cost) * (width - 15))
        y = int((p.pass_rate - min_rate) / (max_rate - min_rate) * (height - 1))
        y = height - 1 - y  # Flip y-axis

        x = max(0, min(x, width - 15))
        y = max(0, min(y, height - 1))

        marker = "*" if p.pareto_optimal else "x"
        label = f" {p.model_name[:12]}"

        # Place marker and label
        grid[y][x] = marker
        for i, ch in enumerate(label):
            if x + 1 + i < width:
                grid[y][x + 1 + i] = ch

    # Render grid with y-axis labels
    for row_idx, row in enumerate(grid):
        rate = max_rate - (row_idx / (height - 1)) * (max_rate - min_rate)
        label = f"{rate:5.0%} |"
        lines.append(label + "".join(row))

    # X-axis
    lines.append("       " + "-" * width)
    x_label = f"       ${min_cost:.3f}" + " " * (width - 20) + f"${max_cost:.3f}"
    lines.append(x_label)
    lines.append(" " * 20 + "Cost per run")

    # Legend
    lines.append("")
    lines.append("* = Pareto-optimal   x = Dominated")

    return "\n".join(lines)


def print_pareto(
    result: ParetoResult,
    console: Any = None,
) -> None:
    """Print Pareto analysis results to terminal.

    Args:
        result: Pareto analysis result.
        console: Rich console (creates one if not provided).
    """
    from rich.console import Console
    from rich.table import Table

    if console is None:
        console = Console()

    # ASCII plot
    console.print(f"\n{render_pareto_ascii(result)}")

    # Table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model", min_width=20)
    table.add_column("Pass Rate", justify="right")
    table.add_column("Cost/Run", justify="right")
    table.add_column("Latency", justify="right")
    table.add_column("Status", justify="center")

    for p in sorted(result.points, key=lambda x: -x.pass_rate):
        rate_style = "green" if p.pass_rate >= 0.8 else "yellow" if p.pass_rate >= 0.5 else "red"
        status = (
            "[green bold]Pareto-optimal[/green bold]"
            if p.pareto_optimal
            else "[dim]Dominated[/dim]"
        )

        table.add_row(
            p.model_name,
            f"[{rate_style}]{p.pass_rate:.0%}[/{rate_style}]",
            f"${p.mean_cost:.4f}",
            f"{p.mean_latency_ms:.0f}ms",
            status,
        )

    console.print(table)

    # Recommendation
    if result.recommendation:
        console.print(f"\n[bold]Recommendation:[/bold] {result.recommendation}")
