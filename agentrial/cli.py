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
@click.option(
    "--update-snapshots",
    is_flag=True,
    help="Save results as new snapshot baseline",
)
@click.option(
    "--judge",
    is_flag=True,
    help="Enable LLM-as-Judge evaluation of agent outputs",
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
    update_snapshots: bool,
    judge: bool,
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
        # Auto-discover: search cwd and tests/ directory for test files
        test_files = discover_test_files(Path.cwd())
        tests_dir = Path.cwd() / "tests"
        if tests_dir.is_dir():
            test_files.extend(discover_test_files(tests_dir))
        test_files = sorted(set(test_files))

    if not test_files:
        console.print("[red]No test files found.[/red]")
        console.print(
            "Agentrial looks for files matching test_*.yml, test_*.yaml, "
            "or test_*.py in the current directory and tests/."
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

        # LLM-as-Judge evaluation
        if judge and not json_output:
            from agentrial.evaluators.llm_judge import LLMJudge, Rubric

            llm_judge = LLMJudge(
                rubric=Rubric(criteria="Evaluate correctness and quality of the agent output."),
                repeats=3,
            )
            console.print("\n[bold]LLM-as-Judge evaluation:[/bold]")
            for eval_result in suite_result.results:
                case_name = eval_result.test_case.name
                # Judge the last trial's output
                last_trial = eval_result.trials[-1] if eval_result.trials else None
                if last_trial:
                    query = eval_result.test_case.input.query
                    judgment = llm_judge.evaluate(query, last_trial.agent_output.output)
                    judge_type = "rule-based" if llm_judge._rule_based else "LLM"
                    style = "green" if judgment.passed else "red"
                    console.print(
                        f"  {case_name}: [{style}]{judgment.mean_score:.1f}/5.0[/{style}] "
                        f"(alpha={judgment.alpha:.2f}, {judge_type})"
                    )
                    if judgment.reasoning_samples:
                        console.print(f"    [dim]{judgment.reasoning_samples[0]}[/dim]")

        # Snapshot handling
        if update_snapshots:
            from agentrial.snapshots import create_snapshot, save_snapshot

            snap = create_snapshot(suite_result)
            snap_path = save_snapshot(snap)
            console.print(f"[green]Snapshot saved: {snap_path}[/green]")
        elif not json_output:
            # Auto-compare with existing snapshot if available
            from agentrial.snapshots import (
                compare_with_snapshot,
                find_snapshot,
                load_snapshot,
                print_snapshot_comparison,
            )

            snap_path = find_snapshot(suite.name)
            if snap_path:
                snap = load_snapshot(snap_path)
                comparison = compare_with_snapshot(suite_result, snap)
                print_snapshot_comparison(comparison, console)
                if not comparison.overall_passed:
                    total_passed = False

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


# --- Snapshot commands ---


@main.group()
def snapshot() -> None:
    """Manage statistical snapshots."""


@snapshot.command("update")
@click.argument("test_path", type=click.Path(exists=True), required=False)
@click.option("-n", "--trials", type=int, default=None, help="Number of trials")
@click.option(
    "-c", "--config", "cfg_path", type=click.Path(exists=True), help="Config file path"
)
@click.option(
    "-o", "--output", "output_path", type=click.Path(), help="Snapshot output path"
)
def snapshot_update(
    test_path: str | None, trials: int | None, cfg_path: str | None,
    output_path: str | None,
) -> None:
    """Run tests and save results as a new snapshot baseline."""
    from agentrial.snapshots import create_snapshot, save_snapshot

    config = load_config(cfg_path)
    effective_trials = trials or config.trials

    test_files = _resolve_test_files(test_path)
    if not test_files:
        console.print("[red]No test files found.[/red]")
        sys.exit(1)

    _ensure_cwd_in_path()
    for test_file in test_files:
        console.print(f"\n[bold]Running:[/bold] {test_file}")
        suite = load_suite(test_file)
        engine = MultiTrialEngine(trials=effective_trials)
        agent = load_agent(suite.agent)
        suite_result = engine.run_suite(agent, suite)
        snap = create_snapshot(suite_result)
        snap_path = save_snapshot(snap, path=output_path)
        console.print(f"[green]Snapshot saved: {snap_path}[/green]")


@snapshot.command("check")
@click.argument("test_path", type=click.Path(exists=True), required=False)
@click.option("-n", "--trials", type=int, default=None, help="Number of trials")
@click.option(
    "-c", "--config", "cfg_path", type=click.Path(exists=True), help="Config file path"
)
def snapshot_check(
    test_path: str | None, trials: int | None, cfg_path: str | None
) -> None:
    """Run tests and compare against existing snapshot."""
    from agentrial.snapshots import (
        compare_with_snapshot,
        find_snapshot,
        load_snapshot,
        print_snapshot_comparison,
    )

    config = load_config(cfg_path)
    effective_trials = trials or config.trials

    test_files = _resolve_test_files(test_path)
    if not test_files:
        console.print("[red]No test files found.[/red]")
        sys.exit(1)

    _ensure_cwd_in_path()
    any_regression = False
    for test_file in test_files:
        console.print(f"\n[bold]Running:[/bold] {test_file}")
        suite = load_suite(test_file)
        engine = MultiTrialEngine(trials=effective_trials)
        agent = load_agent(suite.agent)
        suite_result = engine.run_suite(agent, suite)

        snap_path = find_snapshot(suite.name)
        if not snap_path:
            console.print(
                f"[yellow]No snapshot found for '{suite.name}'. "
                f"Run 'agentrial snapshot update' first.[/yellow]"
            )
            continue

        snap = load_snapshot(snap_path)
        comparison = compare_with_snapshot(suite_result, snap)
        print_snapshot_comparison(comparison, console)
        if not comparison.overall_passed:
            any_regression = True

    sys.exit(1 if any_regression else 0)


# --- Security commands ---


@main.group()
def security() -> None:
    """MCP security scanning."""


@security.command("scan")
@click.option(
    "--mcp-config",
    "mcp_config",
    type=click.Path(exists=True),
    required=True,
    help="Path to MCP config JSON file",
)
@click.option(
    "-o", "--output", "output_path", type=click.Path(), help="Save results as JSON"
)
def security_scan(mcp_config: str, output_path: str | None) -> None:
    """Scan an MCP configuration for security vulnerabilities."""
    import json as json_mod

    from agentrial.security.scanner import print_scan_result, scan_mcp_config

    result = scan_mcp_config(config_path=mcp_config)
    print_scan_result(result, console)

    if output_path:
        report = {
            "score": result.score,
            "passed": result.passed,
            "servers_scanned": result.servers_scanned,
            "tools_scanned": result.tools_scanned,
            "findings": [
                {
                    "severity": f.severity.value,
                    "category": f.category,
                    "title": f.title,
                    "description": f.description,
                    "tool_name": f.tool_name,
                    "server_name": f.server_name,
                }
                for f in result.findings
            ],
        }
        Path(output_path).write_text(json_mod.dumps(report, indent=2))
        console.print(f"[dim]Report saved to {output_path}[/dim]")

    sys.exit(0 if result.passed else 1)


# --- Pareto commands ---


@main.command("pareto")
@click.option(
    "--models",
    required=True,
    help="Comma-separated list of model names",
)
@click.option("-n", "--trials", type=int, default=None, help="Trials per model")
@click.option(
    "-c", "--config", "cfg_path", type=click.Path(exists=True), help="Config file path"
)
@click.argument("test_path", type=click.Path(exists=True), required=False)
def pareto_cmd(
    models: str, trials: int | None, cfg_path: str | None, test_path: str | None
) -> None:
    """Run Pareto frontier analysis across models.

    Evaluates the same test suite with different models and identifies
    which models are Pareto-optimal (best cost-accuracy trade-off).
    """
    from agentrial.pareto import analyze_from_results, print_pareto

    config = load_config(cfg_path)
    effective_trials = trials or config.trials

    model_list = [m.strip() for m in models.split(",")]
    test_files = _resolve_test_files(test_path)
    if not test_files:
        console.print("[red]No test files found.[/red]")
        sys.exit(1)

    _ensure_cwd_in_path()
    test_file = test_files[0]
    suite = load_suite(test_file)

    console.print(
        f"[bold]Pareto analysis:[/bold] {suite.name} with "
        f"{len(model_list)} models x {effective_trials} trials"
    )

    model_results = {}
    for model_name in model_list:
        console.print(f"  Running model: {model_name}...")
        # Inject model name into each test case's context
        for case in suite.cases:
            case.input.context["model"] = model_name
        engine = MultiTrialEngine(trials=effective_trials, show_progress=False)
        agent = load_agent(suite.agent)
        result = engine.run_suite(agent, suite)
        model_results[model_name] = {
            "pass_rate": result.overall_pass_rate,
            "mean_cost": result.total_cost / max(len(result.results), 1),
            "mean_latency_ms": result.total_duration_ms / max(len(result.results), 1),
            "trials": effective_trials,
        }

    pareto_result = analyze_from_results(model_results)
    print_pareto(pareto_result, console)


# --- Prompt commands ---


@main.group()
def prompt() -> None:
    """Prompt version control."""


@prompt.command("track")
@click.argument("prompt_file", type=click.Path(exists=True))
@click.option("--version", "version_name", help="Explicit version name")
@click.option("--tag", multiple=True, help="Tags in key=value format")
def prompt_track(
    prompt_file: str, version_name: str | None, tag: tuple[str, ...]
) -> None:
    """Track a new prompt version from a file."""
    from agentrial.prompts import PromptStore

    text = Path(prompt_file).read_text()
    tags = {}
    for t in tag:
        if "=" in t:
            k, v = t.split("=", 1)
            tags[k] = v

    store = PromptStore()
    pv = store.track(text, version=version_name, tags=tags or None)
    console.print(
        f"[green]Tracked prompt version {pv.version}[/green] "
        f"(hash: {pv.hash})"
    )


@prompt.command("diff")
@click.argument("version_a")
@click.argument("version_b")
def prompt_diff(version_a: str, version_b: str) -> None:
    """Show diff between two prompt versions."""
    from agentrial.prompts import PromptStore, print_prompt_diff

    store = PromptStore()
    try:
        diff = store.diff(version_a, version_b)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)
    print_prompt_diff(diff, console)


@prompt.command("list")
def prompt_list() -> None:
    """List all tracked prompt versions."""
    from agentrial.prompts import PromptStore

    store = PromptStore()
    versions = store.list_versions()
    if not versions:
        console.print("[dim]No prompt versions tracked yet.[/dim]")
        return
    for v in versions:
        pv = store.get_version(v)
        if pv:
            preview = pv.prompt_text[:60].replace("\n", " ")
            console.print(f"  {pv.version}  {pv.hash}  {preview}...")
        else:
            console.print(f"  {v}")


# --- Monitor commands ---


@main.command("monitor")
@click.option(
    "--baseline",
    "baseline_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to baseline snapshot JSON",
)
@click.option("--dry-run", is_flag=True, help="Show detector config without running")
@click.option("--window", type=int, default=50, help="Sliding window size")
@click.option(
    "--cusum-threshold", type=float, default=5.0, help="CUSUM detection threshold"
)
def monitor_cmd(
    baseline_path: str, dry_run: bool, window: int, cusum_threshold: float
) -> None:
    """Configure production drift monitoring from a baseline snapshot."""
    import json as json_mod

    from agentrial.monitor import DriftDetector

    baseline = json_mod.loads(Path(baseline_path).read_text())

    baseline_pass_rate = baseline.get("overall", {}).get("pass_rate", 0.85)

    # Collect baseline cost/latency values from cases
    baseline_costs: list[float] = []
    baseline_latencies: list[float] = []
    for case in baseline.get("cases", []):
        metrics = case.get("metrics", {})
        baseline_costs.extend(metrics.get("cost", {}).get("values", []))
        baseline_latencies.extend(metrics.get("latency", {}).get("values", []))

    _detector = DriftDetector(
        baseline_pass_rate=baseline_pass_rate,
        baseline_costs=baseline_costs or None,
        baseline_latencies=baseline_latencies or None,
        cusum_threshold=cusum_threshold,
        window_size=window,
    )

    if dry_run:
        console.print("[bold]Drift detector configuration:[/bold]")
        console.print(f"  Baseline pass rate: {baseline_pass_rate:.0%}")
        console.print(f"  CUSUM threshold: {cusum_threshold}")
        console.print(f"  Window size: {window}")
        console.print(f"  Baseline costs: {len(baseline_costs)} samples")
        console.print(f"  Baseline latencies: {len(baseline_latencies)} samples")
        console.print("\n[dim]Detector ready. Use the Python API to feed observations.[/dim]")
        return

    console.print(
        f"[bold]Monitor initialized[/bold] "
        f"(baseline: {baseline_pass_rate:.0%}, window: {window})"
    )
    console.print("[dim]Feed observations via the Python API:[/dim]")
    console.print("  from agentrial.monitor import DriftDetector, Observation")
    console.print(f"  detector = DriftDetector(baseline_pass_rate={baseline_pass_rate})")
    console.print("  alerts = detector.observe(Observation(passed=True, cost=0.01))")


# --- Dashboard command ---


@main.command("dashboard")
@click.option("--port", type=int, default=8080, help="Server port")
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option(
    "--data-dir", type=click.Path(), help="Data directory for persistence"
)
def dashboard_cmd(port: int, host: str, data_dir: str | None) -> None:
    """Launch the cloud dashboard web server."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]uvicorn not installed. Install with: "
            "pip install 'agentrial[dashboard]'[/red]"
        )
        sys.exit(1)

    from agentrial.dashboard.app import create_app

    app = create_app(data_dir=data_dir)
    console.print(f"[bold]Starting dashboard at http://{host}:{port}[/bold]")
    uvicorn.run(app, host=host, port=port)


# --- Helper ---


def _resolve_test_files(test_path: str | None) -> list[Path]:
    """Resolve test files from a path argument or auto-discover."""
    if test_path:
        path = Path(test_path)
        if path.is_file():
            return [path]
        return discover_test_files(path)

    test_files = discover_test_files(Path.cwd())
    tests_dir = Path.cwd() / "tests"
    if tests_dir.is_dir():
        test_files.extend(discover_test_files(tests_dir))
    return sorted(set(test_files))


if __name__ == "__main__":
    main()
