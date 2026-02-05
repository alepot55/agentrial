"""Tests for the pytest plugin."""

from __future__ import annotations

from agentrial.pytest_plugin import AgentResult, _AgentTestConfig, _run_agent_test, agent_test
from agentrial.types import AgentInput, AgentMetadata, AgentOutput, StepType, TrajectoryStep


def _make_agent(output: str = "result", tool_name: str = "search"):
    """Create a simple mock agent for testing."""

    def mock_agent(inp: AgentInput) -> AgentOutput:
        return AgentOutput(
            output=output,
            steps=[
                TrajectoryStep(
                    step_index=0,
                    step_type=StepType.TOOL_CALL,
                    name=tool_name,
                    parameters={"query": inp.query},
                    output="tool result",
                    tokens=50,
                ),
            ],
            metadata=AgentMetadata(total_tokens=100, cost=0.01),
        )

    return mock_agent


class TestAgentResult:
    """Tests for AgentResult utility methods."""

    def test_tool_called(self) -> None:
        raw = AgentOutput(
            output="done",
            steps=[
                TrajectoryStep(0, StepType.TOOL_CALL, "search"),
                TrajectoryStep(1, StepType.TOOL_CALL, "analyze"),
            ],
        )
        result = AgentResult(
            output="done",
            steps=list(raw.steps),
            cost=0.01,
            tokens=100,
            duration_ms=500.0,
            success=True,
            error=None,
            pass_rate=1.0,
            trials_passed=1,
            trials_total=1,
            _raw_outputs=[raw],
        )
        assert result.tool_called("search")
        assert result.tool_called("analyze")
        assert not result.tool_called("write")

    def test_tool_called_in_all(self) -> None:
        raw1 = AgentOutput(
            output="done",
            steps=[TrajectoryStep(0, StepType.TOOL_CALL, "search")],
        )
        raw2 = AgentOutput(
            output="done",
            steps=[
                TrajectoryStep(0, StepType.TOOL_CALL, "search"),
                TrajectoryStep(1, StepType.TOOL_CALL, "write"),
            ],
        )
        result = AgentResult(
            output="done",
            steps=[],
            cost=0.01,
            tokens=100,
            duration_ms=500.0,
            success=True,
            error=None,
            pass_rate=1.0,
            trials_passed=2,
            trials_total=2,
            _raw_outputs=[raw1, raw2],
        )
        assert result.tool_called_in_all("search")
        assert not result.tool_called_in_all("write")

    def test_output_contains(self) -> None:
        raw1 = AgentOutput(output="The answer is 42")
        raw2 = AgentOutput(output="Not sure")
        result = AgentResult(
            output="The answer is 42",
            steps=[],
            cost=0.0,
            tokens=0,
            duration_ms=0.0,
            success=True,
            error=None,
            pass_rate=0.5,
            trials_passed=1,
            trials_total=2,
            _raw_outputs=[raw1, raw2],
        )
        assert result.output_contains("42")
        assert not result.output_contains_in_all("42")
        assert result.output_contains_in_all("")  # Empty string is always contained

    def test_mean_cost(self) -> None:
        raw1 = AgentOutput(
            output="a", metadata=AgentMetadata(cost=0.10)
        )
        raw2 = AgentOutput(
            output="b", metadata=AgentMetadata(cost=0.20)
        )
        result = AgentResult(
            output="b",
            steps=[],
            cost=0.15,
            tokens=0,
            duration_ms=0.0,
            success=True,
            error=None,
            pass_rate=1.0,
            trials_passed=2,
            trials_total=2,
            _raw_outputs=[raw1, raw2],
        )
        assert abs(result.mean_cost - 0.15) < 1e-10

    def test_empty_raw_outputs(self) -> None:
        result = AgentResult(
            output="",
            steps=[],
            cost=0.0,
            tokens=0,
            duration_ms=0.0,
            success=True,
            error=None,
            pass_rate=0.0,
            trials_passed=0,
            trials_total=0,
            _raw_outputs=[],
        )
        assert result.mean_cost == 0.0
        assert result.mean_tokens == 0.0
        assert not result.tool_called("anything")
        assert not result.tool_called_in_all("anything")


class TestAgentTestDecorator:
    """Tests for the @agent_test decorator."""

    def test_decorator_stores_config(self) -> None:
        @agent_test(agent="my_module.agent", trials=5, threshold=0.9)
        def test_example(agent_result):
            pass

        config = test_example._agentrial_config
        assert config.agent == "my_module.agent"
        assert config.trials == 5
        assert config.threshold == 0.9

    def test_decorator_defaults(self) -> None:
        @agent_test()
        def test_default(agent_result):
            pass

        config = test_default._agentrial_config
        assert config.trials == 10
        assert config.threshold == 0.8
        assert config.agent is None

    def test_decorator_with_callable_agent(self) -> None:
        mock_agent = _make_agent("hello")

        @agent_test(agent=mock_agent, query="test", trials=2)
        def test_with_fn(agent_result):
            pass

        config = test_with_fn._agentrial_config
        assert config.agent is mock_agent
        assert config.query == "test"


class TestRunAgentTest:
    """Tests for _run_agent_test execution."""

    def test_successful_run(self) -> None:
        mock_agent = _make_agent("The answer is 42", tool_name="search")

        config = _AgentTestConfig(
            agent=mock_agent,
            query="What is the answer?",
            trials=3,
            threshold=0.0,  # No threshold check
        )

        # Track if check_fn was called
        check_called = []

        def check_fn(result: AgentResult) -> None:
            check_called.append(True)
            assert result.output == "The answer is 42"
            assert result.tool_called("search")
            assert result.cost > 0

        _run_agent_test(config, check_fn)
        assert len(check_called) == 1

    def test_threshold_failure(self) -> None:
        import pytest

        # Agent that always fails
        def failing_agent(inp: AgentInput) -> AgentOutput:
            return AgentOutput(output="", success=False, error="always fails")

        config = _AgentTestConfig(
            agent=failing_agent,
            query="test",
            trials=3,
            threshold=0.5,
        )

        def check_fn(result: AgentResult) -> None:
            pass  # Don't assert anything in check_fn

        with pytest.raises(pytest.fail.Exception, match="pass rate"):
            _run_agent_test(config, check_fn)

    def test_no_agent_skips(self) -> None:
        import pytest

        config = _AgentTestConfig(agent=None, query="test")

        def check_fn(result: AgentResult) -> None:
            pass

        with pytest.raises(pytest.skip.Exception):
            _run_agent_test(config, check_fn)

    def test_multiple_trials_metrics(self) -> None:
        call_count = 0

        def counting_agent(inp: AgentInput) -> AgentOutput:
            nonlocal call_count
            call_count += 1
            return AgentOutput(
                output=f"result {call_count}",
                metadata=AgentMetadata(total_tokens=50, cost=0.005),
            )

        config = _AgentTestConfig(
            agent=counting_agent,
            query="test",
            trials=5,
            threshold=0.0,
        )

        results_captured = []

        def check_fn(result: AgentResult) -> None:
            results_captured.append(result)

        _run_agent_test(config, check_fn)

        assert call_count == 5
        assert len(results_captured) == 1
        result = results_captured[0]
        assert result.trials_total == 5
        assert len(result._raw_outputs) == 5


class TestPytestHooks:
    """Tests for pytest CLI hooks."""

    def test_addoption_called(self) -> None:
        from agentrial.pytest_plugin import pytest_addoption

        # Mock parser
        class MockGroup:
            def addoption(self, *args, **kwargs):
                self.options = getattr(self, "options", [])
                self.options.append((args, kwargs))

        class MockParser:
            def __init__(self):
                self.groups = {}

            def getgroup(self, name, desc=""):
                if name not in self.groups:
                    self.groups[name] = MockGroup()
                return self.groups[name]

        parser = MockParser()
        pytest_addoption(parser)

        group = parser.groups["agentrial"]
        assert len(group.options) == 4  # --agenteval, --trials, --agent, --threshold

    def test_configure_marker(self) -> None:
        from agentrial.pytest_plugin import pytest_configure

        class MockConfig:
            def __init__(self):
                self.ini_values = []

            def addinivalue_line(self, name, value):
                self.ini_values.append((name, value))

        config = MockConfig()
        pytest_configure(config)

        assert len(config.ini_values) == 1
        assert config.ini_values[0][0] == "markers"
        assert "agentrial" in config.ini_values[0][1]
