"""Agentrial CLI - Statistical evaluation for AI agents."""

import logging
import os
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console

from agentrial import __version__
from agentrial.config import discover_test_files, load_config, load_suite
from agentrial.reporters.json_report import (
    compare_reports,
    export_json,
    load_json_report,
    save_json_report,
)
from agentrial.reporters.terminal import print_results
from agentrial.runner.engine import MultiTrialEngine, load_agent

console = Console()
logger = logging.getLogger("agentrial")


def _ensure_cwd_in_path() -> None:
    """Ensure current working directory is in Python path for agent imports.

    This allows users to run agentrial without manually setting PYTHONPATH
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
    """Agentrial - Statistical evaluation framework for AI agents.

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
@click.option(
    "--flamegraph",
    is_flag=True,
    help="Show trajectory flame graph for each test case",
)
@click.option(
    "--html",
    "html_path",
    type=click.Path(),
    help="Export flame graph as HTML file",
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
    flamegraph: bool,
    html_path: str | None,
) -> None:
    """Run evaluation suite(s).

    TEST_PATH can be a test file (YAML or Python) or a directory.
    If not specified, looks for test files in current directory.

    Examples:

        agentrial run tests/test_flight_search.yml

        agentrial run --trials 20 --threshold 0.9

        agentrial run -o results.json
    """
    verbose = ctx.obj.get("verbose", False)

    # Load configuration
    cfg_path = Path(config_path) if config_path else None
    config = load_config(cfg_path)

    # Override with CLI options
    effective_trials = trials or config.trials
    _ = threshold or config.threshold  # Used by suite override below

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
        console.print(
            "Agentrial looks for files matching test_*.yml, test_*.yaml, "
            "test_*.py, or agentrial.yml"
        )
        console.print("\nTo get started quickly, run:")
        console.print("  [bold]agentrial init[/bold]  — creates a working sample project")
        console.print("  [bold]agentrial run[/bold]   — runs evaluation")
        sys.exit(1)

    console.print(f"Found {len(test_files)} test file(s)")

    # Run each suite
    all_results = []
    total_passed = True

    for test_file in test_files:
        console.print(f"\n[bold]Running:[/bold] {test_file}")

        try:
            suite = load_suite(test_file)
        except yaml.YAMLError as e:
            console.print(f"[red]YAML parse error in {test_file}:[/red]")
            console.print(f"  {e}")
            console.print("[dim]Hint: check indentation and quoting in your YAML file.[/dim]")
            continue
        except ValueError as e:
            console.print(f"[red]Invalid test file {test_file}: {e}[/red]")
            console.print(
                "[dim]Hint: ensure your YAML has 'suite', 'agent', and 'cases' fields.[/dim]"
            )
            continue
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
            # Provide targeted hints based on the error
            agent_module = suite.agent.rsplit(".", 1)[0] if "." in suite.agent else suite.agent
            console.print("[dim]Troubleshooting:[/dim]")
            if "." not in suite.agent:
                console.print(
                    "  Agent path must be 'module.function' format "
                    f"(got '{suite.agent}')"
                )
            else:
                console.print(f"  1. Is '{agent_module}.py' in your project?")
                console.print("  2. Is the function name spelled correctly?")
                console.print(
                    "  3. Does the module have import errors? "
                    f"Try: python -c \"import {agent_module}\""
                )
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

        # Show flame graph if requested
        if flamegraph and not json_output:
            from agentrial.reporters.flamegraph import print_suite_flamegraphs

            print_suite_flamegraphs(suite_result, console)

        # Export HTML flame graph if requested
        if html_path:
            from agentrial.reporters.flamegraph import export_suite_flamegraphs_html

            html_content = export_suite_flamegraphs_html(suite_result)
            Path(html_path).write_text(html_content)
            console.print(f"[dim]Flame graph saved to {html_path}[/dim]")

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

        agentrial compare results.json --baseline baseline.json
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
            msg = (
                f"  - {r['test_case']}: {r['baseline_pass_rate']:.1%} -> "
                f"{r['current_pass_rate']:.1%} ({r['delta']:+.1%})"
            )
            console.print(msg)
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

        agentrial baseline results.json

        agentrial baseline results.json -o my_baseline.json
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
    """Initialize Agentrial in current directory.

    Creates a sample agent, test suite, and configuration so you can
    run `agentrial run` immediately and see real results.
    """
    # Create sample config
    config_content = """\
# Agentrial configuration
trials: 5
threshold: 0.8
output_format: terminal
verbose: false
"""

    # Create a working sample agent that needs no external dependencies
    agent_content = """\
\"\"\"Sample agent for Agentrial demo.

This is a simple rule-based agent that answers questions about capitals,
math, and greetings. Replace it with your own LLM-based agent.
\"\"\"

import time

from agentrial.types import (
    AgentInput,
    AgentMetadata,
    AgentOutput,
    StepType,
    TrajectoryStep,
)

# Simple knowledge base
CAPITALS = {
    "france": "Paris",
    "italy": "Rome",
    "japan": "Tokyo",
    "germany": "Berlin",
    "spain": "Madrid",
    "uk": "London",
    "united kingdom": "London",
    "usa": "Washington, D.C.",
    "united states": "Washington, D.C.",
    "brazil": "Brasilia",
    "australia": "Canberra",
}


def sample_agent(input: AgentInput) -> AgentOutput:
    \"\"\"A demo agent that answers simple questions.

    Supports:
    - Capital city lookups ("What is the capital of France?")
    - Basic greetings ("Hello", "Hi")
    - Math questions ("What is 2 + 3?")

    Replace this with your own agent (LangGraph, CrewAI, custom, etc.)
    \"\"\"
    query = input.query.lower().strip()
    steps: list[TrajectoryStep] = []
    start = time.time()

    # Step 1: classify intent
    if "capital" in query:
        intent = "capital_lookup"
    elif any(greet in query for greet in ["hello", "hi", "hey", "help"]):
        intent = "greeting"
    elif any(op in query for op in ["+", "-", "*", "/", "plus", "minus", "times"]):
        intent = "math"
    else:
        intent = "general"

    steps.append(
        TrajectoryStep(
            step_index=0,
            step_type=StepType.REASONING,
            name="classify_intent",
            parameters={"query": input.query},
            output=intent,
            duration_ms=1.0,
        )
    )

    # Step 2: generate answer
    if intent == "capital_lookup":
        # Look up capital
        answer = None
        for country, capital in CAPITALS.items():
            if country in query:
                answer = f"The capital of {country.title()} is {capital}."
                break
        if answer is None:
            answer = "I don't know the capital of that country."

        steps.append(
            TrajectoryStep(
                step_index=1,
                step_type=StepType.TOOL_CALL,
                name="lookup_capital",
                parameters={"query": query},
                output=answer,
                duration_ms=2.0,
            )
        )

    elif intent == "greeting":
        answer = (
            "Hello! I'm a demo agent. I can help you with capital city lookups, "
            "basic math, and general questions. How can I help you today?"
        )
        steps.append(
            TrajectoryStep(
                step_index=1,
                step_type=StepType.LLM_CALL,
                name="generate_greeting",
                parameters={},
                output=answer,
                duration_ms=1.0,
            )
        )

    elif intent == "math":
        import re

        nums = re.findall(r"\\d+", query)
        if len(nums) >= 2:
            a, b = int(nums[0]), int(nums[1])
            if "+" in query or "plus" in query:
                result = a + b
            elif "-" in query or "minus" in query:
                result = a - b
            elif "*" in query or "times" in query:
                result = a * b
            elif "/" in query:
                result = a / b if b != 0 else "undefined (division by zero)"
            else:
                result = a + b
            answer = f"The answer is {result}."
        else:
            answer = "I couldn't parse the math expression."

        steps.append(
            TrajectoryStep(
                step_index=1,
                step_type=StepType.TOOL_CALL,
                name="calculate",
                parameters={"expression": query},
                output=answer,
                duration_ms=1.0,
            )
        )

    else:
        answer = "I'm not sure how to answer that. Try asking about capitals or math!"
        steps.append(
            TrajectoryStep(
                step_index=1,
                step_type=StepType.LLM_CALL,
                name="generate_response",
                parameters={"query": query},
                output=answer,
                duration_ms=1.0,
            )
        )

    duration_ms = (time.time() - start) * 1000

    return AgentOutput(
        output=answer,
        steps=steps,
        metadata=AgentMetadata(
            total_tokens=0,
            prompt_tokens=0,
            completion_tokens=0,
            cost=0.0,
            duration_ms=duration_ms,
        ),
        success=True,
    )
"""

    # Create sample test suite that works with the sample agent
    test_content = """\
# Agentrial sample test suite
# Run with: agentrial run
suite: sample-demo
agent: sample_agent.sample_agent
trials: 5
threshold: 0.8

cases:
  - name: greeting
    input:
      query: "Hello, what can you help me with?"
    expected:
      output_contains:
        - "help"

  - name: capital-france
    input:
      query: "What is the capital of France?"
    expected:
      output_contains:
        - "Paris"
      tool_calls:
        - tool: lookup_capital

  - name: capital-japan
    input:
      query: "What is the capital of Japan?"
    expected:
      output_contains:
        - "Tokyo"

  - name: basic-math
    input:
      query: "What is 15 + 27?"
    expected:
      output_contains:
        - "42"
      tool_calls:
        - tool: calculate
"""

    config_path = Path("agentrial.yml")
    agent_path = Path("sample_agent.py")
    test_path = Path("tests/test_sample.yml")

    created_files = []

    if config_path.exists():
        console.print(f"[yellow]Config already exists: {config_path}[/yellow]")
    else:
        config_path.write_text(config_content)
        created_files.append(str(config_path))
        console.print(f"[green]Created config: {config_path}[/green]")

    if agent_path.exists():
        console.print(f"[yellow]Agent already exists: {agent_path}[/yellow]")
    else:
        agent_path.write_text(agent_content)
        created_files.append(str(agent_path))
        console.print(f"[green]Created sample agent: {agent_path}[/green]")

    test_path.parent.mkdir(parents=True, exist_ok=True)
    if test_path.exists():
        console.print(f"[yellow]Test already exists: {test_path}[/yellow]")
    else:
        test_path.write_text(test_content)
        created_files.append(str(test_path))
        console.print(f"[green]Created sample test: {test_path}[/green]")

    if created_files:
        console.print("\n[bold green]Ready to go![/bold green] Run your first evaluation:")
        console.print("  [bold]agentrial run[/bold]")
        console.print(
            "\nThen replace sample_agent.py with your own agent "
            "and update tests/test_sample.yml."
        )
    else:
        console.print("\n[dim]All files already exist. Run: agentrial run[/dim]")


if __name__ == "__main__":
    main()
