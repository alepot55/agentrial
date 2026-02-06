"""Trajectory flame graph visualization for Agentrial.

Provides Rich terminal and HTML export of per-step pass rates across
multiple trials, showing where agent executions diverge and fail.
"""

from __future__ import annotations

import html as html_module
import statistics
from collections import Counter
from dataclasses import dataclass, field

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentrial.types import EvalResult, StepType, SuiteResult


@dataclass
class StepStats:
    """Aggregated statistics for a single step position across trials."""

    step_index: int
    name: str
    step_type: StepType
    total_trials: int = 0
    present_count: int = 0
    pass_count: int = 0
    action_distribution: dict[str, int] = field(default_factory=dict)
    avg_duration_ms: float = 0.0
    avg_tokens: int = 0
    divergence_p: float | None = None

    @property
    def pass_rate(self) -> float:
        if self.present_count == 0:
            return 0.0
        return self.pass_count / self.present_count

    @property
    def presence_rate(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return self.present_count / self.total_trials

    @property
    def is_conditional(self) -> bool:
        """Step is conditional if it doesn't appear in all trials."""
        return self.presence_rate < 1.0 and self.present_count > 0


@dataclass
class FlameGraphData:
    """Data for rendering a trajectory flame graph."""

    test_name: str
    total_trials: int
    pass_count: int
    fail_count: int
    steps: list[StepStats]
    root_cause_step: int | None = None
    root_cause_message: str | None = None
    avg_cost: float = 0.0
    avg_latency_ms: float = 0.0

    @property
    def pass_rate(self) -> float:
        if self.total_trials == 0:
            return 0.0
        return self.pass_count / self.total_trials


def build_flamegraph_data(result: EvalResult) -> FlameGraphData:
    """Build flame graph data from an evaluation result.

    Analyzes all trials to compute per-step pass rates, action distributions,
    and identifies the root cause step where failures diverge.

    Args:
        result: Evaluation result with trial data.

    Returns:
        FlameGraphData ready for rendering.
    """
    trials = result.trials
    total = len(trials)
    passed = sum(1 for t in trials if t.passed)
    failed = total - passed

    # Find the max number of steps across all trials
    max_steps = max((len(t.agent_output.steps) for t in trials), default=0)

    steps: list[StepStats] = []

    for step_idx in range(max_steps):
        # Collect data about this step position across all trials
        step_names: Counter[str] = Counter()
        step_types: Counter[StepType] = Counter()
        durations: list[float] = []
        tokens: list[int] = []
        present_in_passed = 0
        present_in_failed = 0
        present_count = 0

        for trial in trials:
            if step_idx < len(trial.agent_output.steps):
                step = trial.agent_output.steps[step_idx]
                step_names[step.name] += 1
                step_types[step.step_type] += 1
                durations.append(step.duration_ms)
                tokens.append(step.tokens)
                present_count += 1

                if trial.passed:
                    present_in_passed += 1
                else:
                    present_in_failed += 1

        if present_count == 0:
            continue

        # Most common name and type for this step position
        most_common_name = step_names.most_common(1)[0][0]
        most_common_type = step_types.most_common(1)[0][0]

        # Compute pass rate: among trials where this step is present,
        # what fraction of those trials ultimately passed?
        step_pass_count = present_in_passed

        step_stat = StepStats(
            step_index=step_idx,
            name=most_common_name,
            step_type=most_common_type,
            total_trials=total,
            present_count=present_count,
            pass_count=step_pass_count,
            action_distribution=dict(step_names),
            avg_duration_ms=statistics.mean(durations) if durations else 0.0,
            avg_tokens=int(statistics.mean(tokens)) if tokens else 0,
        )

        # Check for divergence using the failure attribution data
        if result.failure_attribution:
            step_divergences = result.failure_attribution.get("step_divergences", [])
            for div in step_divergences:
                if div.get("step_index") == step_idx:
                    step_stat.divergence_p = div.get("p_value")

        steps.append(step_stat)

    # Determine root cause step
    root_cause_step = None
    root_cause_message = None
    if result.failure_attribution:
        most_likely = result.failure_attribution.get("most_likely_step")
        if most_likely is not None:
            root_cause_step = most_likely
        root_cause_message = result.failure_attribution.get("recommendation")

    costs = [t.cost for t in trials]
    latencies = [t.duration_ms for t in trials]

    return FlameGraphData(
        test_name=result.test_case.name,
        total_trials=total,
        pass_count=passed,
        fail_count=failed,
        steps=steps,
        root_cause_step=root_cause_step,
        root_cause_message=root_cause_message,
        avg_cost=statistics.mean(costs) if costs else 0.0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0.0,
    )


def _bar(rate: float, width: int = 20) -> str:
    """Create a text-based progress bar."""
    filled = int(rate * width)
    empty = width - filled
    return "\u2588" * filled + "\u2591" * empty


def _pct(rate: float) -> str:
    """Format as percentage."""
    return f"{rate * 100:.0f}%"


def _rate_style(rate: float) -> str:
    """Get Rich style for a pass rate."""
    if rate >= 0.9:
        return "green"
    elif rate >= 0.7:
        return "yellow"
    return "red"


def print_flamegraph(data: FlameGraphData, console: Console | None = None) -> None:
    """Render a trajectory flame graph to the terminal using Rich.

    Args:
        data: Flame graph data to render.
        console: Rich console to use (creates new one if not provided).
    """
    if console is None:
        console = Console()

    # Header
    if data.pass_rate >= 0.7:
        status_style = "green"
    elif data.pass_rate >= 0.5:
        status_style = "yellow"
    else:
        status_style = "red"
    header = (
        f"[bold]{data.test_name}[/bold]  "
        f"[{status_style}]PASS {data.pass_count}/{data.total_trials}, "
        f"{_pct(data.pass_rate)}[/{status_style}]"
    )

    # Build step rows
    table = Table(
        show_header=True,
        header_style="bold",
        box=None,
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("Step", min_width=25, no_wrap=True)
    table.add_column("Pass Rate", min_width=25, no_wrap=True)
    table.add_column("Count", justify="right", min_width=12)
    table.add_column("Note", min_width=20)

    for step in data.steps:
        # Step name with type indicator
        type_icon = {
            StepType.TOOL_CALL: "T",
            StepType.LLM_CALL: "L",
            StepType.REASONING: "R",
            StepType.OBSERVATION: "O",
            StepType.OUTPUT: ">",
        }.get(step.step_type, "?")

        name_text = f"Step {step.step_index}: [{type_icon}] {step.name}"
        if step.is_conditional:
            name_text += " *"

        # Bar
        style = _rate_style(step.pass_rate)
        bar_text = Text()
        bar_text.append(_bar(step.pass_rate), style=style)
        bar_text.append(f"  {_pct(step.pass_rate)}", style=style)

        # Count
        count_text = f"({step.pass_count}/{step.present_count})"

        # Note
        note = ""
        if step.divergence_p is not None and step.divergence_p < 0.05:
            note = f"[red]divergence p={step.divergence_p:.2f}[/red]"
        elif step.is_conditional:
            note = f"[dim]conditional ({_pct(step.presence_rate)} of runs)[/dim]"

        if data.root_cause_step == step.step_index:
            note = "[bold red]<-- root cause[/bold red]"

        table.add_row(name_text, bar_text, count_text, note)

    # Root cause footer
    footer_parts = []
    if data.root_cause_message:
        footer_parts.append(f"[bold]Root cause:[/bold] {data.root_cause_message}")
    footer_parts.append(
        f"[dim]Cost: ${data.avg_cost:.4f} avg | Latency: {data.avg_latency_ms:.0f}ms avg[/dim]"
    )
    footer = "\n".join(footer_parts)

    panel_content = table
    console.print(
        Panel(
            panel_content,
            title=header,
            subtitle=footer,
            border_style="blue",
        )
    )


def print_suite_flamegraphs(suite_result: SuiteResult, console: Console | None = None) -> None:
    """Print flame graphs for all test cases in a suite that have failures.

    Args:
        suite_result: Suite result to visualize.
        console: Rich console to use.
    """
    if console is None:
        console = Console()

    for result in suite_result.results:
        # Only show flame graph for tests with mixed results (some pass, some fail)
        # or tests that have trajectory steps worth visualizing
        if result.pass_rate < 1.0 or len(result.trials) > 0:
            data = build_flamegraph_data(result)
            if data.steps:
                print_flamegraph(data, console)
                console.print()


def export_flamegraph_html(data: FlameGraphData) -> str:
    """Export flame graph as a standalone HTML page.

    The generated HTML has no external dependencies and can be shared
    as a single file or embedded in PR comments.

    Args:
        data: Flame graph data to render.

    Returns:
        Complete HTML string.
    """
    esc = html_module.escape

    # Build step rows
    step_rows = []
    for step in data.steps:
        rate = step.pass_rate
        pct = f"{rate * 100:.0f}%"
        bar_width = int(rate * 100)

        if rate >= 0.9:
            color = "#22c55e"
        elif rate >= 0.7:
            color = "#eab308"
        else:
            color = "#ef4444"

        type_label = {
            StepType.TOOL_CALL: "tool",
            StepType.LLM_CALL: "llm",
            StepType.REASONING: "reason",
            StepType.OBSERVATION: "obs",
            StepType.OUTPUT: "output",
        }.get(step.step_type, "?")

        note = ""
        if data.root_cause_step == step.step_index:
            note = '<span style="color:#ef4444;font-weight:bold">root cause</span>'
        elif step.divergence_p is not None and step.divergence_p < 0.05:
            note = f'<span style="color:#ef4444">divergence p={step.divergence_p:.2f}</span>'
        elif step.is_conditional:
            note = f'<span style="color:#888">conditional ({_pct(step.presence_rate)})</span>'

        # Action distribution
        actions_html = ""
        if len(step.action_distribution) > 1:
            action_parts = []
            for action, count in sorted(
                step.action_distribution.items(), key=lambda x: -x[1]
            ):
                action_parts.append(f"{esc(action)}: {count}")
            actions_html = (
                f'<div style="font-size:11px;color:#888;margin-top:2px">'
                f'{", ".join(action_parts)}</div>'
            )

        step_rows.append(f"""
        <tr>
          <td style="padding:6px 12px;white-space:nowrap">
            <span style="color:#888">Step {step.step_index}:</span>
            <span style="background:#f0f0f0;padding:1px 6px;border-radius:3px;
                   font-size:11px;color:#666">{esc(type_label)}</span>
            <strong>{esc(step.name)}</strong>
            {actions_html}
          </td>
          <td style="padding:6px 12px;width:200px">
            <div style="background:#e5e7eb;border-radius:4px;height:20px;position:relative">
              <div style="background:{color};border-radius:4px;height:20px;width:{bar_width}%">
              </div>
              <span style="position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);
                     font-size:12px;font-weight:bold;color:#333">{pct}</span>
            </div>
          </td>
          <td style="padding:6px 12px;text-align:right;color:#666">
            {step.pass_count}/{step.present_count}
          </td>
          <td style="padding:6px 12px">{note}</td>
        </tr>""")

    # Overall status
    if data.pass_rate >= 0.7:
        status_color = "#22c55e"
    elif data.pass_rate >= 0.5:
        status_color = "#eab308"
    else:
        status_color = "#ef4444"

    root_cause_html = ""
    if data.root_cause_message:
        root_cause_html = f"""
    <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:12px;
         margin-top:16px">
      <strong>Root cause:</strong> {esc(data.root_cause_message)}
    </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Agentrial: {esc(data.test_name)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
           margin: 40px auto; max-width: 800px; color: #333; background: #fafafa; }}
    table {{ border-collapse: collapse; width: 100%; }}
    tr:hover {{ background: #f9fafb; }}
    th {{ text-align: left; padding: 8px 12px; border-bottom: 2px solid #e5e7eb;
         font-size: 12px; text-transform: uppercase; color: #666; }}
  </style>
</head>
<body>
  <h2 style="margin-bottom:4px">{esc(data.test_name)}</h2>
  <p style="color:{status_color};font-size:18px;font-weight:bold;margin-top:0">
    PASS {data.pass_count}/{data.total_trials} ({_pct(data.pass_rate)})
  </p>
  <p style="color:#888;font-size:14px">
    Cost: ${data.avg_cost:.4f} avg &middot; Latency: {data.avg_latency_ms:.0f}ms avg
  </p>

  <table>
    <tr>
      <th>Step</th>
      <th>Pass Rate</th>
      <th>Count</th>
      <th>Note</th>
    </tr>
    {''.join(step_rows)}
  </table>

  {root_cause_html}

  <p style="color:#bbb;font-size:11px;margin-top:24px;text-align:center">
    Generated by <a href="https://github.com/alepot55/agentrial" style="color:#999">agentrial</a>
  </p>
</body>
</html>"""


def export_suite_flamegraphs_html(suite_result: SuiteResult) -> str:
    """Export flame graphs for all test cases as a single HTML page.

    Args:
        suite_result: Suite result to export.

    Returns:
        Complete HTML string with all flame graphs.
    """
    sections = []
    for result in suite_result.results:
        data = build_flamegraph_data(result)
        if data.steps:
            sections.append(export_flamegraph_html(data))

    if not sections:
        return "<html><body><p>No trajectory data to display.</p></body></html>"

    # For a multi-test page, extract just the body content from each
    # and wrap in a single page
    esc = html_module.escape
    bodies = []
    for result in suite_result.results:
        data = build_flamegraph_data(result)
        if not data.steps:
            continue
        # Re-render as a section (simplified)
        single = export_flamegraph_html(data)
        # Extract between <body> and </body>
        start = single.find("<body>") + len("<body>")
        end = single.find("</body>")
        if start > 0 and end > 0:
            bodies.append(single[start:end])

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Agentrial: {esc(suite_result.suite.name)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
           margin: 40px auto; max-width: 800px; color: #333; background: #fafafa; }}
    table {{ border-collapse: collapse; width: 100%; }}
    tr:hover {{ background: #f9fafb; }}
    th {{ text-align: left; padding: 8px 12px; border-bottom: 2px solid #e5e7eb;
         font-size: 12px; text-transform: uppercase; color: #666; }}
    .section {{ margin-bottom: 48px; border-bottom: 1px solid #e5e7eb;
               padding-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>{esc(suite_result.suite.name)}</h1>
  {'<div class="section">'.join(bodies)}
  <p style="color:#bbb;font-size:11px;margin-top:24px;text-align:center">
    Generated by <a href="https://github.com/alepot55/agentrial" style="color:#999">agentrial</a>
  </p>
</body>
</html>"""
