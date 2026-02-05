"""AgentEval CLI - Statistical evaluation for AI agents."""

import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console

from agenteval import __version__
from agenteval.config import Config, discover_test_files, load_config, load_suite
from agenteval.reporters.json_report import (
    compare_reports,
    export_json,
    load_json_report,
    save_json_report,
)
from agenteval.reporters.terminal import print_comparison, print_results
from agenteval.runner.engine import MultiTrialEngine, load_agent

console = Console()
logger = logging.getLogger("agenteval")


def _ensure_cwd_in_path() -> None:
    """Ensure current working directory is in Python path for agent imports.

    This allows users to run agenteval without manually setting PYTHONPATH
    when their agent module is in the current directory.
    """
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
        logger.debug(f"Added {cwd} to sys.path for agent discovery")


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """AgentEval - Statistical evaluation framework for AI agents.

    The pytest for agent trajectories. Test your agents with multi-trial
    execution, confidence intervals, and step-level failure attribution.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose

    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
    )


@main.command()
@click.argument("test_path", type=click.Path(exists=True), required=False)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True),
    help="Path to config file",
)
@click.option(
    "-n",
    "--trials",
    type=int,
    help="Number of trials per test case (overrides config)",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    help="Minimum pass rate threshold (overrides config)",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(),
    help="Path to save JSON report",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Output results as JSON to stdout",
)
@click.pass_context
def run(
    ctx: click.Context,
    test_path: str | None,
    config_path: str | None,
    trials: int | None,
    threshold: float | None,
    output_path: str | None,
    json_output: bool,
) -> None:
    """Run evaluation suite(s).

    TEST_PATH can be a test file (YAML or Python) or a directory.
    If not specified, looks for test files in current directory.

    Examples:

        agenteval run tests/test_flight_search.yml

        agenteval run --trials 20 --threshold 0.9

        agenteval run -o results.json
    """
    verbose = ctx.obj.get("verbose", False)

    # Load configuration
    cfg_path = Path(config_path) if config_path else None
    config = load_config(cfg_path)

    # Override with CLI options
    effective_trials = trials or config.trials
    effective_threshold = threshold or config.threshold

    # Find test files
    if test_path:
        path = Path(test_path)
        if path.is_file():
            test_files = [path]
        else:
            test_files = discover_test_files(path)
    else:
        test_files = discover_test_files(Path.cwd())

    if not test_files:
        console.print("[red]No test files found.[/red]")
        console.print("Create a test file (test_*.yml or test_*.py) or specify a path.")
        sys.exit(1)

    console.print(f"Found {len(test_files)} test file(s)")

    # Run each suite
    all_results = []
    total_passed = True

    for test_file in test_files:
        console.print(f"\n[bold]Running:[/bold] {test_file}")

        try:
            suite = load_suite(test_file)
        except Exception as e:
            console.print(f"[red]Error loading {test_file}: {e}[/red]")
            continue

        # Override suite settings
        if threshold:
            suite.threshold = threshold

        # Create engine and run
        # Disable progress bar for JSON output to avoid interfering with stdout
        engine = MultiTrialEngine(trials=effective_trials, show_progress=not json_output)

        # Ensure cwd is in path for agent imports
        _ensure_cwd_in_path()

        try:
            agent = load_agent(suite.agent)
        except ImportError as e:
            console.print(f"[red]Error loading agent '{suite.agent}': {e}[/red]")
            continue

        suite_result = engine.run_suite(agent, suite)
        all_results.append(suite_result)

        if not suite_result.passed:
            total_passed = False

        # Print results
        if json_output:
            import json

            print(json.dumps(export_json(suite_result), indent=2))
        else:
            print_results(suite_result, verbose=verbose)

        # Save JSON report if requested
        if output_path:
            save_json_report(suite_result, output_path)
            console.print(f"[dim]Report saved to {output_path}[/dim]")

    # Exit with appropriate code
    sys.exit(0 if total_passed else 1)


@main.command()
@click.argument("current_path", type=click.Path(exists=True))
@click.option(
    "-b",
    "--baseline",
    "baseline_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to baseline JSON report",
)
@click.pass_context
def compare(
    ctx: click.Context,
    current_path: str,
    baseline_path: str,
) -> None:
    """Compare current results against a baseline.

    Detects statistically significant regressions in pass rate.

    Examples:

        agenteval compare results.json --baseline baseline.json
    """
    current = load_json_report(current_path)
    baseline = load_json_report(baseline_path)

    comparison = compare_reports(current, baseline)

    # Print comparison
    console.print(f"\n[bold]Comparison: {current_path} vs {baseline_path}[/bold]")
    console.print(f"Overall delta: {comparison['overall_delta']:+.1%}")

    if comparison["current_passed"] and not comparison["baseline_passed"]:
        console.print("[green]Status improved from FAIL to PASS[/green]")
    elif not comparison["current_passed"] and comparison["baseline_passed"]:
        console.print("[red]Status regressed from PASS to FAIL[/red]")

    # Check for regressions
    regressions = [c for c in comparison["comparisons"] if c["status"] == "regression"]
    if regressions:
        console.print(f"\n[red]Found {len(regressions)} regression(s):[/red]")
        for r in regressions:
            console.print(
                f"  - {r['test_case']}: {r['baseline_pass_rate']:.1%} -> {r['current_pass_rate']:.1%} ({r['delta']:+.1%})"
            )
        sys.exit(1)

    console.print("[green]No regressions detected.[/green]")
    sys.exit(0)


@main.command()
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(),
    default="baseline.json",
    help="Path to save baseline (default: baseline.json)",
)
@click.argument("results_path", type=click.Path(exists=True))
def baseline(results_path: str, output_path: str) -> None:
    """Save results as a new baseline.

    Use this to establish a baseline for regression detection.

    Examples:

        agenteval baseline results.json

        agenteval baseline results.json -o my_baseline.json
    """
    import shutil

    shutil.copy(results_path, output_path)
    console.print(f"[green]Baseline saved to {output_path}[/green]")


@main.command("config")
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True),
    help="Path to config file",
)
def show_config(config_path: str | None) -> None:
    """Show current configuration.

    Displays merged configuration from file and defaults.
    """
    cfg_path = Path(config_path) if config_path else None
    config = load_config(cfg_path)

    console.print("[bold]Current Configuration[/bold]")
    console.print(f"  trials: {config.trials}")
    console.print(f"  threshold: {config.threshold}")
    console.print(f"  output_format: {config.output_format}")
    console.print(f"  parallel: {config.parallel}")
    console.print(f"  verbose: {config.verbose}")


@main.command()
def init() -> None:
    """Initialize AgentEval in current directory.

    Creates a sample configuration and test file to get started.
    """
    # Create sample config
    config_content = """# AgentEval configuration
trials: 10
threshold: 0.85
output_format: terminal
verbose: false
"""

    # Create sample test
    test_content = """# Sample AgentEval test suite
suite: sample-agent
agent: my_module.my_agent  # Update this to your agent's import path
trials: 10
threshold: 0.85

cases:
  - name: basic-query
    input:
      query: "Hello, what can you help me with?"
    expected:
      output_contains:
        - "help"
"""

    config_path = Path("agenteval.yml")
    test_path = Path("tests/test_sample.yml")

    if config_path.exists():
        console.print(f"[yellow]Config file already exists: {config_path}[/yellow]")
    else:
        config_path.write_text(config_content)
        console.print(f"[green]Created config: {config_path}[/green]")

    test_path.parent.mkdir(parents=True, exist_ok=True)
    if test_path.exists():
        console.print(f"[yellow]Test file already exists: {test_path}[/yellow]")
    else:
        test_path.write_text(test_content)
        console.print(f"[green]Created sample test: {test_path}[/green]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Update agenteval.yml with your settings")
    console.print("2. Edit tests/test_sample.yml with your agent path and test cases")
    console.print("3. Run: agenteval run")


if __name__ == "__main__":
    main()
