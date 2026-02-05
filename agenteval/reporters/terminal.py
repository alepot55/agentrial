"""Rich terminal output for AgentEval results."""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agenteval.types import EvalResult, SuiteResult

console = Console()


def format_percentage(value: float) -> str:
    """Format a value as a percentage."""
    return f"{value * 100:.1f}%"


def format_ci(lower: float, upper: float) -> str:
    """Format a confidence interval."""
    return f"({format_percentage(lower)}-{format_percentage(upper)})"


def format_cost(value: float) -> str:
    """Format a cost value."""
    if value < 0.01:
        return f"${value:.4f}"
    return f"${value:.3f}"


def format_latency(value: float) -> str:
    """Format latency in milliseconds."""
    if value < 1000:
        return f"{value:.0f}ms"
    return f"{value / 1000:.2f}s"


def get_pass_rate_style(pass_rate: float, threshold: float) -> str:
    """Get style based on pass rate vs threshold."""
    if pass_rate >= threshold:
        return "green"
    elif pass_rate >= threshold * 0.8:
        return "yellow"
    return "red"


def print_results(suite_result: SuiteResult, verbose: bool = False) -> None:
    """Print suite results to terminal.

    Args:
        suite_result: The suite result to print.
        verbose: Whether to print detailed information.
    """
    # Print header
    console.print()
    status = "[green]PASSED[/green]" if suite_result.passed else "[red]FAILED[/red]"
    console.print(
        Panel(
            f"[bold]{suite_result.suite.name}[/bold] - {status}",
            subtitle=f"Threshold: {format_percentage(suite_result.suite.threshold)}",
        )
    )

    # Create results table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Test Case", style="cyan")
    table.add_column("Pass Rate", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Avg Latency", justify="right")
    table.add_column("Avg Tokens", justify="right")

    for result in suite_result.results:
        # Determine style based on pass rate
        style = get_pass_rate_style(result.pass_rate, suite_result.suite.threshold)

        table.add_row(
            result.test_case.name,
            Text(format_percentage(result.pass_rate), style=style),
            format_ci(result.pass_rate_ci.lower, result.pass_rate_ci.upper),
            format_cost(result.mean_cost),
            format_latency(result.mean_latency_ms),
            str(int(result.mean_tokens)),
        )

    console.print(table)

    # Print summary
    console.print()
    summary_style = "green" if suite_result.passed else "red"
    console.print(
        f"[bold]Overall Pass Rate:[/bold] "
        f"[{summary_style}]{format_percentage(suite_result.overall_pass_rate)}[/{summary_style}] "
        f"{format_ci(suite_result.overall_pass_rate_ci.lower, suite_result.overall_pass_rate_ci.upper)}"
    )
    console.print(f"[bold]Total Cost:[/bold] {format_cost(suite_result.total_cost)}")
    console.print(f"[bold]Total Duration:[/bold] {format_latency(suite_result.total_duration_ms)}")
    console.print()

    # Print failure attribution if verbose
    if verbose:
        _print_failure_attribution(suite_result)


def _print_failure_attribution(suite_result: SuiteResult) -> None:
    """Print failure attribution details."""
    has_failures = False

    for result in suite_result.results:
        if result.failure_attribution and result.failure_attribution.get("most_likely_step") is not None:
            has_failures = True
            console.print(f"\n[bold yellow]Failure Analysis: {result.test_case.name}[/bold yellow]")
            console.print(f"  {result.failure_attribution.get('recommendation', 'No recommendation')}")

            # Print failure patterns
            patterns = result.failure_attribution.get("failure_patterns", [])
            if patterns:
                console.print("  [bold]Common failures:[/bold]")
                for pattern in patterns[:3]:
                    console.print(
                        f"    - {pattern['message'][:80]}... ({pattern['count']} occurrences)"
                    )

    if not has_failures:
        console.print("[dim]No significant failure patterns detected.[/dim]")


def print_eval_result(result: EvalResult, verbose: bool = False) -> None:
    """Print a single test case result.

    Args:
        result: The evaluation result to print.
        verbose: Whether to print detailed information.
    """
    console.print()
    console.print(f"[bold cyan]{result.test_case.name}[/bold cyan]")

    # Pass rate with CI
    style = get_pass_rate_style(result.pass_rate, 0.85)
    console.print(
        f"  Pass Rate: [{style}]{format_percentage(result.pass_rate)}[/{style}] "
        f"{format_ci(result.pass_rate_ci.lower, result.pass_rate_ci.upper)}"
    )

    # Metrics
    console.print(f"  Cost: {format_cost(result.mean_cost)} (avg per trial)")
    console.print(f"  Latency: {format_latency(result.mean_latency_ms)} (avg)")
    console.print(f"  Tokens: {int(result.mean_tokens)} (avg)")

    if verbose and result.failure_attribution:
        recommendation = result.failure_attribution.get("recommendation", "")
        if recommendation:
            console.print(f"  [yellow]Analysis: {recommendation}[/yellow]")


def print_comparison(
    current: SuiteResult,
    baseline: SuiteResult,
    alpha: float = 0.05,
) -> None:
    """Print comparison between current and baseline results.

    Args:
        current: Current run results.
        baseline: Baseline results to compare against.
        alpha: Significance level for regression detection.
    """
    from agenteval.metrics.statistical import detect_regression

    console.print()
    console.print(Panel("[bold]Comparison with Baseline[/bold]"))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Test Case")
    table.add_column("Current", justify="right")
    table.add_column("Baseline", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status")

    # Match test cases by name
    baseline_by_name = {r.test_case.name: r for r in baseline.results}

    for current_result in current.results:
        baseline_result = baseline_by_name.get(current_result.test_case.name)
        if baseline_result is None:
            table.add_row(
                current_result.test_case.name,
                format_percentage(current_result.pass_rate),
                "-",
                "-",
                "[blue]NEW[/blue]",
            )
            continue

        # Detect regression
        current_passed = sum(1 for t in current_result.trials if t.passed)
        baseline_passed = sum(1 for t in baseline_result.trials if t.passed)

        regression = detect_regression(
            current_passed,
            len(current_result.trials),
            baseline_passed,
            len(baseline_result.trials),
            alpha,
        )

        delta = regression["rate_delta"]
        delta_str = f"{delta:+.1%}"

        if regression["is_regression"]:
            status = f"[red]REGRESSION (p={regression['p_value']:.3f})[/red]"
            delta_style = "red"
        elif delta > 0 and regression["p_value"] < alpha:
            status = f"[green]IMPROVED (p={regression['p_value']:.3f})[/green]"
            delta_style = "green"
        else:
            status = "[dim]no change[/dim]"
            delta_style = "dim"

        table.add_row(
            current_result.test_case.name,
            format_percentage(current_result.pass_rate),
            format_percentage(baseline_result.pass_rate),
            Text(delta_str, style=delta_style),
            status,
        )

    console.print(table)
