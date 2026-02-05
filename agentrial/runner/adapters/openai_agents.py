"""OpenAI Agents SDK adapter for Agentrial.

Wraps OpenAI Agents SDK agents to capture trajectory and produce
standardized AgentOutput. The SDK is imported only at runtime so the
core package never depends on it.

Usage:
    from agentrial.runner.adapters.openai_agents import wrap_openai_agent
    agent_fn = wrap_openai_agent(my_agent)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from agentrial.runner.adapters.base import BaseAdapter
from agentrial.runner.trajectory import TrajectoryRecorder
from agentrial.types import AgentInput, AgentOutput, StepType


def _import_openai_agents() -> Any:
    """Import openai agents SDK at runtime."""
    try:
        import agents  # noqa: F811

        return agents
    except ImportError as err:
        raise ImportError(
            "OpenAI Agents SDK is not installed. Install it with: "
            "pip install openai-agents"
        ) from err


class OpenAIAgentsAdapter(BaseAdapter):
    """Adapter for OpenAI Agents SDK.

    Wraps an openai-agents Agent and captures tool calls, handoffs,
    and model responses from the run result.
    """

    def __init__(
        self,
        agent: Any,
        context: Any = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            agent: An agents.Agent instance.
            context: Optional context to pass to Runner.run_sync().
        """
        self.agent = agent
        self.context = context

    def __call__(self, input: AgentInput) -> AgentOutput:
        """Execute the agent and capture trajectory.

        Args:
            input: Agent input with query and context.

        Returns:
            Standardized AgentOutput.
        """
        agents_sdk = _import_openai_agents()
        recorder = TrajectoryRecorder()

        try:
            start = time.time()

            # Use Runner.run_sync for synchronous execution
            result = agents_sdk.Runner.run_sync(
                self.agent,
                input.query,
                context=self.context,
            )
            duration = (time.time() - start) * 1000

            output_text = str(result.final_output)

            # Extract trajectory from run result
            self._extract_trajectory(result, recorder)

            agent_output = recorder.finalize(output_text)
            agent_output.metadata.duration_ms = duration
            return agent_output

        except Exception as e:
            return recorder.finalize(
                output="",
                success=False,
                error=f"OpenAI Agents execution failed: {e}",
            )

    def _extract_trajectory(
        self, result: Any, recorder: TrajectoryRecorder
    ) -> None:
        """Extract trajectory steps from an OpenAI Agents run result."""
        # The result contains new_items with the execution trace
        if not hasattr(result, "new_items"):
            return

        total_input_tokens = 0
        total_output_tokens = 0

        for item in result.new_items:
            item_type = getattr(item, "type", "")

            if item_type == "tool_call_item":
                # Tool was called
                call = getattr(item, "raw_item", item)
                recorder.add_step(
                    step_type=StepType.TOOL_CALL,
                    name=getattr(call, "name", "tool_call"),
                    parameters={"arguments": getattr(call, "arguments", "")},
                    metadata={
                        "call_id": getattr(call, "call_id", ""),
                        "agent_id": getattr(item, "agent", {}).get(
                            "name", "unknown"
                        )
                        if isinstance(getattr(item, "agent", None), dict)
                        else str(getattr(getattr(item, "agent", None), "name", "unknown")),
                    },
                )

            elif item_type == "tool_call_output_item":
                # Tool output
                recorder.add_step(
                    step_type=StepType.OBSERVATION,
                    name="tool_output",
                    output=getattr(item, "output", ""),
                )

            elif item_type == "message_output_item":
                # Model response message
                raw = getattr(item, "raw_item", item)
                content = ""
                if hasattr(raw, "content"):
                    for part in raw.content:
                        if hasattr(part, "text"):
                            content += part.text

                agent_name = "unknown"
                agent_attr = getattr(item, "agent", None)
                if agent_attr is not None:
                    if isinstance(agent_attr, dict):
                        agent_name = agent_attr.get("name", "unknown")
                    else:
                        agent_name = getattr(agent_attr, "name", "unknown")

                recorder.add_step(
                    step_type=StepType.LLM_CALL,
                    name=f"response_{agent_name}",
                    output=content,
                    metadata={"agent_id": agent_name},
                )

            elif item_type == "handoff_item":
                # Agent handoff
                source = getattr(item, "source_agent", "unknown")
                target = getattr(item, "target_agent", "unknown")
                if hasattr(source, "name"):
                    source = source.name
                if hasattr(target, "name"):
                    target = target.name

                recorder.add_step(
                    step_type=StepType.OBSERVATION,
                    name="handoff",
                    parameters={"from": str(source), "to": str(target)},
                    output=f"Handoff from {source} to {target}",
                    metadata={
                        "agent_id": str(source),
                        "handoff_target": str(target),
                    },
                )

        # Extract usage from result
        if hasattr(result, "raw_responses"):
            for resp in result.raw_responses:
                usage = getattr(resp, "usage", None)
                if usage:
                    total_input_tokens += getattr(
                        usage, "input_tokens", 0
                    ) or getattr(usage, "prompt_tokens", 0)
                    total_output_tokens += getattr(
                        usage, "output_tokens", 0
                    ) or getattr(usage, "completion_tokens", 0)

        if total_input_tokens or total_output_tokens:
            recorder.add_token_usage(
                prompt_tokens=total_input_tokens,
                completion_tokens=total_output_tokens,
            )

    def get_agent_callable(self) -> Callable[[AgentInput], AgentOutput]:
        """Get the adapter as a callable."""
        return self


def wrap_openai_agent(
    agent: Any,
    context: Any = None,
) -> Callable[[AgentInput], AgentOutput]:
    """Wrap an OpenAI Agents SDK agent as an Agentrial-compatible agent.

    Args:
        agent: An agents.Agent instance from openai-agents.
        context: Optional context for the runner.

    Returns:
        Callable that takes AgentInput and returns AgentOutput.

    Example:
        from agents import Agent
        agent = Agent(name="assistant", instructions="Be helpful.")
        agent_fn = wrap_openai_agent(agent)
    """
    return OpenAIAgentsAdapter(agent, context)
