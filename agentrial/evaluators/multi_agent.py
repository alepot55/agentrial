"""Multi-agent evaluation metrics.

Evaluates interactions between agents in multi-agent systems:
- Delegation accuracy (orchestrator -> correct sub-agent)
- Handoff fidelity (context preserved between agents)
- Redundancy detection (duplicate work across agents)
- Cascade failure detection (failure propagation depth)
- Communication efficiency (tokens spent on inter-agent messages)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentrial.types import AgentOutput, StepType, TrajectoryStep


@dataclass
class AgentStep(TrajectoryStep):
    """A trajectory step annotated with the agent that executed it.

    Extends TrajectoryStep with agent_id to track which agent in a
    multi-agent system performed this step.
    """

    agent_id: str = "default"


@dataclass
class MultiAgentMetrics:
    """Metrics for a multi-agent evaluation run."""

    agents_involved: list[str]
    delegation_accuracy: float | None = None
    handoff_fidelity: float | None = None
    redundancy_rate: float = 0.0
    redundant_calls: list[dict[str, Any]] = field(default_factory=list)
    cascade_depth: int = 0
    communication_tokens: int = 0
    total_tokens: int = 0
    communication_efficiency: float = 0.0


@dataclass
class MultiAgentExpectation:
    """Expected behavior for a multi-agent test case.

    Attributes:
        expected_agents: Which agents should participate.
        delegation_correct: Whether delegation should be correct.
        max_redundant_calls: Maximum acceptable redundant tool calls.
        max_total_tokens: Maximum total tokens across all agents.
        handoff_preserves: Keys that must be preserved across handoffs.
    """

    expected_agents: list[str] | None = None
    delegation_correct: bool | None = None
    max_redundant_calls: int | None = None
    max_total_tokens: int | None = None
    handoff_preserves: list[str] | None = None


def extract_agent_ids(steps: list[TrajectoryStep]) -> list[str]:
    """Extract unique agent IDs from trajectory steps.

    Looks for agent_id in step metadata or uses AgentStep.agent_id
    if available.

    Args:
        steps: List of trajectory steps.

    Returns:
        List of unique agent IDs in order of first appearance.
    """
    seen: dict[str, None] = {}
    for step in steps:
        if isinstance(step, AgentStep):
            agent_id = step.agent_id
        else:
            agent_id = step.metadata.get("agent_id", "default")
        if agent_id not in seen:
            seen[agent_id] = None
    return list(seen.keys())


def compute_delegation_accuracy(
    steps: list[TrajectoryStep],
    expected_agents: list[str],
) -> float:
    """Compute delegation accuracy.

    Measures whether the orchestrator delegated to the correct sub-agents.

    Args:
        steps: Trajectory steps with agent_id metadata.
        expected_agents: Expected list of agents that should participate.

    Returns:
        Fraction of expected agents that were actually involved (0-1).
    """
    actual_agents = set(extract_agent_ids(steps))
    if not expected_agents:
        return 1.0

    matched = sum(1 for a in expected_agents if a in actual_agents)
    return matched / len(expected_agents)


def compute_redundancy(steps: list[TrajectoryStep]) -> tuple[float, list[dict[str, Any]]]:
    """Detect redundant work across agents.

    A tool call is redundant if the same tool with the same parameters
    is called by different agents.

    Args:
        steps: Trajectory steps with agent_id metadata.

    Returns:
        Tuple of (redundancy_rate, list of redundant call details).
    """
    tool_calls: list[tuple[str, str, str]] = []  # (agent_id, tool_name, params_key)

    for step in steps:
        if step.step_type != StepType.TOOL_CALL:
            continue

        if isinstance(step, AgentStep):
            agent_id = step.agent_id
        else:
            agent_id = step.metadata.get("agent_id", "default")

        # Create a hashable key for the tool call parameters
        params_key = str(sorted(step.parameters.items())) if step.parameters else ""
        tool_calls.append((agent_id, step.name, params_key))

    if not tool_calls:
        return 0.0, []

    # Find duplicate tool+params combinations across different agents
    call_map: dict[tuple[str, str], list[str]] = {}
    for agent_id, tool_name, params_key in tool_calls:
        key = (tool_name, params_key)
        if key not in call_map:
            call_map[key] = []
        call_map[key].append(agent_id)

    redundant = []
    total_redundant = 0
    for (tool_name, _params_key), agents in call_map.items():
        unique_agents = set(agents)
        if len(unique_agents) > 1:
            # Same call made by multiple agents
            extra = len(agents) - 1
            total_redundant += extra
            redundant.append({
                "tool": tool_name,
                "agents": list(unique_agents),
                "total_calls": len(agents),
                "redundant_calls": extra,
            })

    rate = total_redundant / len(tool_calls) if tool_calls else 0.0
    return rate, redundant


def compute_cascade_depth(
    steps: list[TrajectoryStep],
) -> int:
    """Detect cascade failure depth in multi-agent execution.

    Counts how many consecutive agent handoffs result in failures.

    Args:
        steps: Trajectory steps with agent_id metadata.

    Returns:
        Maximum depth of cascading failures.
    """
    if not steps:
        return 0

    agents_in_order: list[str] = []
    agent_failed: dict[str, bool] = {}

    for step in steps:
        if isinstance(step, AgentStep):
            agent_id = step.agent_id
        else:
            agent_id = step.metadata.get("agent_id", "default")

        if not agents_in_order or agents_in_order[-1] != agent_id:
            agents_in_order.append(agent_id)

        # Track if agent had a failure (output indicates error)
        if step.output and isinstance(step.output, str):
            if any(err in step.output.lower() for err in ["error", "failed", "exception"]):
                agent_failed[agent_id] = True

    # Count consecutive failing agents
    max_depth = 0
    current_depth = 0
    for agent in agents_in_order:
        if agent_failed.get(agent, False):
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        else:
            current_depth = 0

    return max_depth


def compute_communication_efficiency(
    steps: list[TrajectoryStep],
) -> tuple[int, int, float]:
    """Compute inter-agent communication efficiency.

    Measures the ratio of tokens spent on inter-agent messages vs total tokens.

    Args:
        steps: Trajectory steps with agent_id metadata.

    Returns:
        Tuple of (communication_tokens, total_tokens, efficiency_ratio).
    """
    total_tokens = 0
    comm_tokens = 0
    prev_agent = None

    for step in steps:
        if isinstance(step, AgentStep):
            agent_id = step.agent_id
        else:
            agent_id = step.metadata.get("agent_id", "default")

        total_tokens += step.tokens

        # Tokens at agent boundaries are "communication overhead"
        if prev_agent is not None and agent_id != prev_agent:
            comm_tokens += step.tokens

        prev_agent = agent_id

    efficiency = 1.0 - (comm_tokens / total_tokens) if total_tokens > 0 else 1.0
    return comm_tokens, total_tokens, efficiency


def evaluate_multi_agent(
    output: AgentOutput,
    expectations: MultiAgentExpectation | None = None,
) -> tuple[MultiAgentMetrics, list[str]]:
    """Evaluate a multi-agent execution.

    Args:
        output: Agent output with trajectory steps.
        expectations: Optional expectations to validate against.

    Returns:
        Tuple of (metrics, failure_messages).
    """
    steps = output.steps
    failures: list[str] = []

    # Extract agents
    agents = extract_agent_ids(steps)

    # Delegation accuracy
    delegation_acc = None
    if expectations and expectations.expected_agents:
        delegation_acc = compute_delegation_accuracy(steps, expectations.expected_agents)
        if expectations.delegation_correct and delegation_acc < 1.0:
            missing = set(expectations.expected_agents) - set(agents)
            failures.append(
                f"Delegation incorrect: missing agents {missing}"
            )

    # Redundancy
    redundancy_rate, redundant_calls = compute_redundancy(steps)
    if expectations and expectations.max_redundant_calls is not None:
        total_redundant = sum(r["redundant_calls"] for r in redundant_calls)
        if total_redundant > expectations.max_redundant_calls:
            failures.append(
                f"Redundancy exceeded: {total_redundant} redundant calls "
                f"(max {expectations.max_redundant_calls})"
            )

    # Cascade depth
    cascade = compute_cascade_depth(steps)

    # Communication efficiency
    comm_tokens, total_tokens, efficiency = compute_communication_efficiency(steps)

    # Token limit
    if expectations and expectations.max_total_tokens is not None:
        if total_tokens > expectations.max_total_tokens:
            failures.append(
                f"Token limit exceeded: {total_tokens} "
                f"(max {expectations.max_total_tokens})"
            )

    metrics = MultiAgentMetrics(
        agents_involved=agents,
        delegation_accuracy=delegation_acc,
        redundancy_rate=redundancy_rate,
        redundant_calls=redundant_calls,
        cascade_depth=cascade,
        communication_tokens=comm_tokens,
        total_tokens=total_tokens,
        communication_efficiency=efficiency,
    )

    return metrics, failures
