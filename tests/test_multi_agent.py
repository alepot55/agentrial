"""Tests for multi-agent evaluation."""

from agentrial.evaluators.multi_agent import (
    AgentStep,
    MultiAgentExpectation,
    compute_cascade_depth,
    compute_communication_efficiency,
    compute_delegation_accuracy,
    compute_redundancy,
    evaluate_multi_agent,
    extract_agent_ids,
)
from agentrial.types import (
    AgentMetadata,
    AgentOutput,
    StepType,
    TrajectoryStep,
)


def _agent_step(
    name: str,
    agent_id: str,
    step_type: StepType = StepType.TOOL_CALL,
    step_index: int = 0,
    tokens: int = 100,
    output: str = "ok",
    **params: str,
) -> AgentStep:
    """Create an AgentStep for testing."""
    return AgentStep(
        step_index=step_index,
        step_type=step_type,
        name=name,
        parameters=dict(params),
        output=output,
        tokens=tokens,
        agent_id=agent_id,
    )


def _meta_step(
    name: str,
    agent_id: str,
    step_type: StepType = StepType.TOOL_CALL,
    step_index: int = 0,
    tokens: int = 100,
    output: str = "ok",
    **params: str,
) -> TrajectoryStep:
    """Create a TrajectoryStep with agent_id in metadata."""
    return TrajectoryStep(
        step_index=step_index,
        step_type=step_type,
        name=name,
        parameters=dict(params),
        output=output,
        tokens=tokens,
        metadata={"agent_id": agent_id},
    )


class TestExtractAgentIds:
    """Tests for extract_agent_ids."""

    def test_from_agent_steps(self) -> None:
        steps = [
            _agent_step("search", "researcher"),
            _agent_step("analyze", "analyst"),
            _agent_step("write", "writer"),
        ]
        agents = extract_agent_ids(steps)
        assert agents == ["researcher", "analyst", "writer"]

    def test_from_metadata(self) -> None:
        steps = [
            _meta_step("search", "researcher"),
            _meta_step("write", "writer"),
        ]
        agents = extract_agent_ids(steps)
        assert agents == ["researcher", "writer"]

    def test_deduplication(self) -> None:
        steps = [
            _agent_step("step1", "agent-a"),
            _agent_step("step2", "agent-b"),
            _agent_step("step3", "agent-a"),
        ]
        agents = extract_agent_ids(steps)
        assert agents == ["agent-a", "agent-b"]

    def test_default_agent(self) -> None:
        steps = [TrajectoryStep(0, StepType.TOOL_CALL, "search")]
        agents = extract_agent_ids(steps)
        assert agents == ["default"]


class TestDelegationAccuracy:
    """Tests for compute_delegation_accuracy."""

    def test_perfect_delegation(self) -> None:
        steps = [
            _agent_step("search", "researcher"),
            _agent_step("analyze", "analyst"),
        ]
        acc = compute_delegation_accuracy(steps, ["researcher", "analyst"])
        assert acc == 1.0

    def test_partial_delegation(self) -> None:
        steps = [
            _agent_step("search", "researcher"),
        ]
        acc = compute_delegation_accuracy(steps, ["researcher", "analyst"])
        assert acc == 0.5

    def test_no_expected(self) -> None:
        steps = [_agent_step("search", "researcher")]
        acc = compute_delegation_accuracy(steps, [])
        assert acc == 1.0


class TestRedundancy:
    """Tests for compute_redundancy."""

    def test_no_redundancy(self) -> None:
        steps = [
            _agent_step("search", "agent-a", query="q1"),
            _agent_step("analyze", "agent-b", data="d1"),
        ]
        rate, redundant = compute_redundancy(steps)
        assert rate == 0.0
        assert len(redundant) == 0

    def test_same_call_different_agents(self) -> None:
        steps = [
            _agent_step("search", "agent-a", query="q1"),
            _agent_step("search", "agent-b", query="q1"),
        ]
        rate, redundant = compute_redundancy(steps)
        assert rate > 0
        assert len(redundant) == 1
        assert redundant[0]["tool"] == "search"
        assert redundant[0]["redundant_calls"] == 1

    def test_same_call_same_agent_not_redundant(self) -> None:
        steps = [
            _agent_step("search", "agent-a", query="q1"),
            _agent_step("search", "agent-a", query="q1"),
        ]
        rate, redundant = compute_redundancy(steps)
        # Same agent calling same tool is not cross-agent redundancy
        assert len(redundant) == 0

    def test_non_tool_steps_ignored(self) -> None:
        steps = [
            _agent_step("think", "agent-a", step_type=StepType.REASONING),
            _agent_step("think", "agent-b", step_type=StepType.REASONING),
        ]
        rate, redundant = compute_redundancy(steps)
        assert rate == 0.0


class TestCascadeDepth:
    """Tests for compute_cascade_depth."""

    def test_no_failures(self) -> None:
        steps = [
            _agent_step("search", "agent-a", output="found results"),
            _agent_step("analyze", "agent-b", output="analysis complete"),
        ]
        depth = compute_cascade_depth(steps)
        assert depth == 0

    def test_single_failure(self) -> None:
        steps = [
            _agent_step("search", "agent-a", output="error: timeout"),
            _agent_step("analyze", "agent-b", output="ok"),
        ]
        depth = compute_cascade_depth(steps)
        assert depth == 1

    def test_cascade_failure(self) -> None:
        steps = [
            _agent_step("search", "agent-a", output="error: not found"),
            _agent_step("retry", "agent-b", output="failed: no data"),
            _agent_step("fallback", "agent-c", output="exception raised"),
        ]
        depth = compute_cascade_depth(steps)
        assert depth == 3

    def test_recovery_breaks_cascade(self) -> None:
        steps = [
            _agent_step("search", "agent-a", output="error: timeout"),
            _agent_step("retry", "agent-b", output="success"),
            _agent_step("analyze", "agent-c", output="failed analysis"),
        ]
        depth = compute_cascade_depth(steps)
        assert depth == 1  # Cascade broken by agent-b success


class TestCommunicationEfficiency:
    """Tests for compute_communication_efficiency."""

    def test_single_agent(self) -> None:
        steps = [
            _agent_step("step1", "agent-a", tokens=100),
            _agent_step("step2", "agent-a", tokens=100),
        ]
        comm, total, efficiency = compute_communication_efficiency(steps)
        assert comm == 0
        assert total == 200
        assert efficiency == 1.0

    def test_multi_agent_handoff(self) -> None:
        steps = [
            _agent_step("step1", "agent-a", tokens=100),
            _agent_step("step2", "agent-b", tokens=50),  # Handoff
            _agent_step("step3", "agent-b", tokens=100),
        ]
        comm, total, efficiency = compute_communication_efficiency(steps)
        assert comm == 50  # Only the first step of agent-b after handoff
        assert total == 250
        assert efficiency == 1.0 - 50 / 250

    def test_empty_steps(self) -> None:
        comm, total, efficiency = compute_communication_efficiency([])
        assert comm == 0
        assert total == 0
        assert efficiency == 1.0


class TestEvaluateMultiAgent:
    """Tests for the evaluate_multi_agent function."""

    def test_passing_evaluation(self) -> None:
        steps = [
            _agent_step("search", "researcher", step_index=0, tokens=100),
            _agent_step("analyze", "analyst", step_index=1, tokens=100),
            _agent_step("write", "writer", step_index=2, tokens=100),
        ]
        output = AgentOutput(
            output="Report complete",
            steps=steps,
            metadata=AgentMetadata(total_tokens=300),
        )

        expectations = MultiAgentExpectation(
            expected_agents=["researcher", "analyst", "writer"],
            delegation_correct=True,
            max_redundant_calls=0,
            max_total_tokens=500,
        )

        metrics, failures = evaluate_multi_agent(output, expectations)
        assert len(failures) == 0
        assert metrics.delegation_accuracy == 1.0
        assert metrics.redundancy_rate == 0.0
        assert set(metrics.agents_involved) == {"researcher", "analyst", "writer"}

    def test_missing_agent_fails(self) -> None:
        steps = [
            _agent_step("search", "researcher"),
        ]
        output = AgentOutput(output="Partial", steps=steps)

        expectations = MultiAgentExpectation(
            expected_agents=["researcher", "analyst"],
            delegation_correct=True,
        )

        metrics, failures = evaluate_multi_agent(output, expectations)
        assert len(failures) == 1
        assert "missing agents" in failures[0].lower()
        assert metrics.delegation_accuracy == 0.5

    def test_redundancy_fails(self) -> None:
        steps = [
            _agent_step("search", "agent-a", query="q"),
            _agent_step("search", "agent-b", query="q"),
        ]
        output = AgentOutput(output="Done", steps=steps)

        expectations = MultiAgentExpectation(max_redundant_calls=0)

        metrics, failures = evaluate_multi_agent(output, expectations)
        assert len(failures) == 1
        assert "redundancy" in failures[0].lower()

    def test_token_limit_fails(self) -> None:
        steps = [
            _agent_step("step1", "agent-a", tokens=500),
            _agent_step("step2", "agent-b", tokens=600),
        ]
        output = AgentOutput(output="Done", steps=steps)

        expectations = MultiAgentExpectation(max_total_tokens=1000)

        metrics, failures = evaluate_multi_agent(output, expectations)
        assert len(failures) == 1
        assert "token limit" in failures[0].lower()

    def test_no_expectations(self) -> None:
        steps = [_agent_step("search", "agent-a")]
        output = AgentOutput(output="Done", steps=steps)

        metrics, failures = evaluate_multi_agent(output, None)
        assert len(failures) == 0
        assert metrics.agents_involved == ["agent-a"]
