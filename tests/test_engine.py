"""Tests for the execution engine."""

import pytest

from agenteval.runner.engine import MultiTrialEngine
from agenteval.types import (
    AgentInput,
    AgentMetadata,
    AgentOutput,
    ExpectedOutput,
    StepType,
    Suite,
    TestCase,
    TrajectoryStep,
)


def create_simple_agent(pass_rate: float = 1.0):
    """Create a simple test agent that passes at a given rate."""
    call_count = [0]

    def agent(input: AgentInput) -> AgentOutput:
        call_count[0] += 1
        should_pass = (call_count[0] % int(1 / pass_rate)) != 0 if pass_rate < 1.0 else True

        return AgentOutput(
            output=f"Response to: {input.query}" if should_pass else "Error response",
            steps=[
                TrajectoryStep(
                    step_index=0,
                    step_type=StepType.TOOL_CALL,
                    name="search",
                    parameters={"query": input.query},
                    output="search results",
                    tokens=50,
                ),
                TrajectoryStep(
                    step_index=1,
                    step_type=StepType.OUTPUT,
                    name="response",
                    output="final response",
                    tokens=30,
                ),
            ],
            metadata=AgentMetadata(
                total_tokens=80,
                cost=0.001,
                duration_ms=100.0,
            ),
            success=True,
        )

    return agent


class TestMultiTrialEngine:
    """Tests for MultiTrialEngine."""

    def test_run_single_trial(self) -> None:
        """Test running a single trial."""
        engine = MultiTrialEngine(trials=1)
        agent = create_simple_agent()

        test_case = TestCase(
            name="test",
            input=AgentInput(query="test query"),
            expected=ExpectedOutput(contains=["Response"]),
        )

        result = engine.run_single_trial(agent, test_case, 0)

        assert result.passed
        assert result.trial_index == 0
        assert result.tokens == 80
        assert result.cost == 0.001

    def test_run_single_trial_failure(self) -> None:
        """Test trial that fails expectation."""
        engine = MultiTrialEngine(trials=1)
        agent = create_simple_agent()

        test_case = TestCase(
            name="test",
            input=AgentInput(query="test query"),
            expected=ExpectedOutput(contains=["not present"]),
        )

        result = engine.run_single_trial(agent, test_case, 0)

        assert not result.passed
        assert len(result.failures) > 0

    def test_run_test_case(self) -> None:
        """Test running all trials for a test case."""
        engine = MultiTrialEngine(trials=5)
        agent = create_simple_agent()

        test_case = TestCase(
            name="test",
            input=AgentInput(query="test query"),
            expected=ExpectedOutput(contains=["Response"]),
        )

        eval_result = engine.run_test_case(agent, test_case)

        assert eval_result.pass_rate == 1.0
        assert len(eval_result.trials) == 5
        assert eval_result.pass_rate_ci.lower > 0.5
        assert eval_result.mean_cost > 0

    def test_run_test_case_partial_failure(self) -> None:
        """Test test case with some failures."""
        engine = MultiTrialEngine(trials=10)

        # Agent that alternates pass/fail
        call_count = [0]

        def alternating_agent(input: AgentInput) -> AgentOutput:
            call_count[0] += 1
            output = "good response" if call_count[0] % 2 == 0 else "bad"
            return AgentOutput(
                output=output,
                metadata=AgentMetadata(cost=0.001, duration_ms=50),
            )

        test_case = TestCase(
            name="test",
            input=AgentInput(query="test"),
            expected=ExpectedOutput(contains=["good"]),
        )

        eval_result = engine.run_test_case(alternating_agent, test_case)

        # Should be around 50% pass rate
        assert 0.3 < eval_result.pass_rate < 0.7
        assert eval_result.pass_rate_ci.lower < 0.5
        assert eval_result.pass_rate_ci.upper > 0.5

    def test_run_suite(self) -> None:
        """Test running a full suite."""
        engine = MultiTrialEngine(trials=3)
        agent = create_simple_agent()

        suite = Suite(
            name="test-suite",
            agent="unused",  # We pass agent directly
            trials=3,
            threshold=0.8,
            cases=[
                TestCase(
                    name="case1",
                    input=AgentInput(query="query 1"),
                    expected=ExpectedOutput(contains=["Response"]),
                ),
                TestCase(
                    name="case2",
                    input=AgentInput(query="query 2"),
                    expected=ExpectedOutput(contains=["Response"]),
                ),
            ],
        )

        suite_result = engine.run_suite(agent, suite)

        assert suite_result.passed
        assert suite_result.overall_pass_rate == 1.0
        assert len(suite_result.results) == 2
        assert suite_result.total_cost > 0

    def test_cost_constraint(self) -> None:
        """Test that cost constraint is enforced."""
        engine = MultiTrialEngine(trials=1)

        def expensive_agent(input: AgentInput) -> AgentOutput:
            return AgentOutput(
                output="expensive result",
                metadata=AgentMetadata(cost=1.0),
            )

        test_case = TestCase(
            name="test",
            input=AgentInput(query="test"),
            max_cost=0.10,
        )

        result = engine.run_single_trial(expensive_agent, test_case, 0)

        assert not result.passed
        assert any("cost" in f.lower() for f in result.failures)

    def test_latency_constraint(self) -> None:
        """Test that latency constraint is enforced."""
        engine = MultiTrialEngine(trials=1)

        def slow_agent(input: AgentInput) -> AgentOutput:
            import time

            time.sleep(0.1)  # 100ms
            return AgentOutput(
                output="slow result",
                metadata=AgentMetadata(duration_ms=100),
            )

        test_case = TestCase(
            name="test",
            input=AgentInput(query="test"),
            max_latency_ms=50,
        )

        result = engine.run_single_trial(slow_agent, test_case, 0)

        assert not result.passed
        assert any("latency" in f.lower() for f in result.failures)

    def test_agent_exception_handling(self) -> None:
        """Test that agent exceptions are handled gracefully."""
        engine = MultiTrialEngine(trials=1)

        def failing_agent(input: AgentInput) -> AgentOutput:
            raise ValueError("Agent error!")

        test_case = TestCase(
            name="test",
            input=AgentInput(query="test"),
        )

        result = engine.run_single_trial(failing_agent, test_case, 0)

        assert not result.passed
        assert any("exception" in f.lower() for f in result.failures)
