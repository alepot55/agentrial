"""Tests for trajectory flame graph."""

from agentrial.reporters.flamegraph import (
    FlameGraphData,
    StepStats,
    build_flamegraph_data,
    export_flamegraph_html,
    print_flamegraph,
)
from agentrial.types import (
    AgentInput,
    AgentMetadata,
    AgentOutput,
    ConfidenceInterval,
    EvalResult,
    StepType,
    TestCase,
    TrajectoryStep,
    TrialResult,
)


def _make_trial(
    passed: bool,
    steps: list[TrajectoryStep] | None = None,
    cost: float = 0.01,
    duration_ms: float = 100.0,
    trial_index: int = 0,
) -> TrialResult:
    """Create a trial result for testing."""
    if steps is None:
        steps = [
            TrajectoryStep(
                step_index=0,
                step_type=StepType.REASONING,
                name="plan",
                duration_ms=10.0,
            ),
            TrajectoryStep(
                step_index=1,
                step_type=StepType.TOOL_CALL,
                name="search",
                duration_ms=50.0,
            ),
        ]

    tc = TestCase(name="test", input=AgentInput(query="q"))
    return TrialResult(
        trial_index=trial_index,
        test_case=tc,
        agent_output=AgentOutput(
            output="result",
            steps=steps,
            metadata=AgentMetadata(cost=cost, duration_ms=duration_ms),
        ),
        passed=passed,
        duration_ms=duration_ms,
        cost=cost,
    )


def _make_eval_result(
    pass_count: int = 7,
    fail_count: int = 3,
) -> EvalResult:
    """Create an eval result with mixed pass/fail trials."""
    trials = []
    for i in range(pass_count):
        trials.append(_make_trial(passed=True, trial_index=i))
    for i in range(fail_count):
        # Failed trials have different step 1 (wrong tool)
        fail_steps = [
            TrajectoryStep(
                step_index=0,
                step_type=StepType.REASONING,
                name="plan",
                duration_ms=10.0,
            ),
            TrajectoryStep(
                step_index=1,
                step_type=StepType.TOOL_CALL,
                name="wrong_tool",
                duration_ms=50.0,
            ),
        ]
        trials.append(
            _make_trial(
                passed=False,
                steps=fail_steps,
                trial_index=pass_count + i,
            )
        )

    total = pass_count + fail_count
    tc = TestCase(name="mixed-test", input=AgentInput(query="q"))
    return EvalResult(
        test_case=tc,
        trials=trials,
        pass_rate=pass_count / total,
        pass_rate_ci=ConfidenceInterval(lower=0.4, upper=0.9),
        mean_cost=0.01,
        cost_ci=ConfidenceInterval(lower=0.005, upper=0.015),
        mean_latency_ms=100.0,
        latency_ci=ConfidenceInterval(lower=80.0, upper=120.0),
        mean_tokens=0.0,
        failure_attribution={
            "most_likely_step": 1,
            "recommendation": "Step 1 (search) diverges: 'wrong_tool' in failed runs",
            "step_divergences": [
                {"step_index": 1, "p_value": 0.03},
            ],
        },
    )


class TestBuildFlameGraphData:
    """Tests for build_flamegraph_data."""

    def test_basic_structure(self) -> None:
        result = _make_eval_result(pass_count=7, fail_count=3)
        data = build_flamegraph_data(result)

        assert data.test_name == "mixed-test"
        assert data.total_trials == 10
        assert data.pass_count == 7
        assert data.fail_count == 3
        assert data.pass_rate == 0.7
        assert len(data.steps) == 2

    def test_step_stats(self) -> None:
        result = _make_eval_result(pass_count=7, fail_count=3)
        data = build_flamegraph_data(result)

        # Step 0: "plan" - present in all trials, all passed trials have it
        step0 = data.steps[0]
        assert step0.name == "plan"
        assert step0.step_type == StepType.REASONING
        assert step0.present_count == 10
        assert step0.total_trials == 10
        assert step0.pass_count == 7  # 7 passing trials

        # Step 1: mixed between "search" and "wrong_tool"
        step1 = data.steps[1]
        assert step1.present_count == 10
        assert step1.pass_count == 7

    def test_action_distribution(self) -> None:
        result = _make_eval_result(pass_count=7, fail_count=3)
        data = build_flamegraph_data(result)

        # Step 0: all "plan"
        assert data.steps[0].action_distribution == {"plan": 10}

        # Step 1: mixed
        assert data.steps[1].action_distribution["search"] == 7
        assert data.steps[1].action_distribution["wrong_tool"] == 3

    def test_root_cause_from_attribution(self) -> None:
        result = _make_eval_result()
        data = build_flamegraph_data(result)

        assert data.root_cause_step == 1
        assert "wrong_tool" in (data.root_cause_message or "")

    def test_divergence_p_value(self) -> None:
        result = _make_eval_result()
        data = build_flamegraph_data(result)

        # Step 1 should have divergence p-value from attribution
        assert data.steps[1].divergence_p == 0.03

    def test_all_passing(self) -> None:
        result = _make_eval_result(pass_count=10, fail_count=0)
        # Override failure_attribution since all pass
        result.failure_attribution = None
        data = build_flamegraph_data(result)

        assert data.pass_rate == 1.0
        assert data.root_cause_step is None
        for step in data.steps:
            assert step.pass_rate == 1.0

    def test_conditional_step(self) -> None:
        """Test steps that don't appear in all trials."""
        trials = []
        for i in range(5):
            # Some trials have 3 steps, some have 2
            steps = [
                TrajectoryStep(0, StepType.REASONING, "plan"),
                TrajectoryStep(1, StepType.TOOL_CALL, "search"),
            ]
            if i < 3:
                steps.append(TrajectoryStep(2, StepType.OUTPUT, "format"))
            trials.append(_make_trial(passed=True, steps=steps, trial_index=i))

        tc = TestCase(name="cond-test", input=AgentInput(query="q"))
        result = EvalResult(
            test_case=tc,
            trials=trials,
            pass_rate=1.0,
            pass_rate_ci=ConfidenceInterval(lower=0.6, upper=1.0),
            mean_cost=0.01,
            cost_ci=ConfidenceInterval(lower=0.005, upper=0.015),
            mean_latency_ms=100.0,
            latency_ci=ConfidenceInterval(lower=80.0, upper=120.0),
            mean_tokens=0.0,
        )

        data = build_flamegraph_data(result)
        assert len(data.steps) == 3

        # Step 2 should be conditional
        assert data.steps[2].is_conditional
        assert data.steps[2].present_count == 3
        assert data.steps[2].presence_rate == 0.6


class TestPrintFlamegraph:
    """Tests for terminal rendering (smoke tests)."""

    def test_renders_without_error(self) -> None:
        from io import StringIO

        from rich.console import Console

        result = _make_eval_result()
        data = build_flamegraph_data(result)

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        print_flamegraph(data, console)

        output = buf.getvalue()
        assert "mixed-test" in output
        assert "plan" in output

    def test_renders_all_passing(self) -> None:
        from io import StringIO

        from rich.console import Console

        result = _make_eval_result(pass_count=10, fail_count=0)
        result.failure_attribution = None
        data = build_flamegraph_data(result)

        buf = StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        print_flamegraph(data, console)

        output = buf.getvalue()
        assert "PASS 10/10" in output


class TestExportHtml:
    """Tests for HTML export."""

    def test_produces_valid_html(self) -> None:
        result = _make_eval_result()
        data = build_flamegraph_data(result)
        html = export_flamegraph_html(data)

        assert "<!DOCTYPE html>" in html
        assert "mixed-test" in html
        assert "agentrial" in html
        assert "plan" in html
        assert "root cause" in html.lower()

    def test_html_escapes_names(self) -> None:
        """Ensure HTML special chars in step names are escaped."""
        data = FlameGraphData(
            test_name="test<script>",
            total_trials=1,
            pass_count=1,
            fail_count=0,
            steps=[
                StepStats(
                    step_index=0,
                    name="tool<b>bold</b>",
                    step_type=StepType.TOOL_CALL,
                    total_trials=1,
                    present_count=1,
                    pass_count=1,
                ),
            ],
        )
        html = export_flamegraph_html(data)

        assert "<script>" not in html
        assert "&lt;script&gt;" in html
        assert "<b>bold</b>" not in html
