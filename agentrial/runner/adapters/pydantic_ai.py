"""Pydantic AI adapter for Agentrial.

Wraps Pydantic AI agents to capture trajectory and produce
standardized AgentOutput. Pydantic AI is imported only at runtime
so the core package never depends on it.

Usage:
    from agentrial.runner.adapters.pydantic_ai import wrap_pydantic_ai_agent
    agent_fn = wrap_pydantic_ai_agent(my_agent)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from agentrial.runner.adapters.base import BaseAdapter
from agentrial.runner.trajectory import TrajectoryRecorder
from agentrial.types import AgentInput, AgentOutput, StepType


def _import_pydantic_ai() -> Any:
    """Import pydantic_ai at runtime."""
    try:
        import pydantic_ai  # noqa: F811

        return pydantic_ai
    except ImportError as err:
        raise ImportError(
            "Pydantic AI is not installed. Install it with: pip install pydantic-ai"
        ) from err


class PydanticAIAdapter(BaseAdapter):
    """Adapter for Pydantic AI agents.

    Wraps a pydantic_ai.Agent and captures tool calls, model
    requests, and token usage from the run result.
    """

    def __init__(
        self,
        agent: Any,
        deps: Any = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            agent: A pydantic_ai.Agent instance.
            deps: Optional dependencies to pass to the agent.
        """
        self.agent = agent
        self.deps = deps

    def __call__(self, input: AgentInput) -> AgentOutput:
        """Execute the agent and capture trajectory.

        Args:
            input: Agent input with query and context.

        Returns:
            Standardized AgentOutput.
        """
        recorder = TrajectoryRecorder()

        try:
            start = time.time()

            # Pydantic AI agents use run_sync for synchronous execution
            result = self.agent.run_sync(
                input.query,
                deps=self.deps,
            )
            duration = (time.time() - start) * 1000

            output_text = str(result.data)

            # Extract trajectory from result messages
            self._extract_trajectory(result, recorder)

            # Extract token usage
            if hasattr(result, "usage"):
                usage = result.usage()
                recorder.add_token_usage(
                    prompt_tokens=getattr(usage, "request_tokens", 0),
                    completion_tokens=getattr(usage, "response_tokens", 0),
                )

            # Extract cost
            if hasattr(result, "cost"):
                cost = result.cost()
                if hasattr(cost, "total"):
                    recorder.add_token_usage(cost=cost.total)

            agent_output = recorder.finalize(output_text)
            agent_output.metadata.duration_ms = duration
            return agent_output

        except Exception as e:
            return recorder.finalize(
                output="",
                success=False,
                error=f"Pydantic AI execution failed: {e}",
            )

    def _extract_trajectory(
        self, result: Any, recorder: TrajectoryRecorder
    ) -> None:
        """Extract trajectory steps from a Pydantic AI result."""
        # Pydantic AI stores messages in result.all_messages()
        if not hasattr(result, "all_messages"):
            return

        messages = result.all_messages()
        for msg in messages:
            msg_kind = getattr(msg, "kind", "")

            if msg_kind == "request":
                # Model request — contains parts (system, user, tool-return)
                parts = getattr(msg, "parts", [])
                for part in parts:
                    part_kind = getattr(part, "part_kind", "")
                    if part_kind == "tool-return":
                        recorder.add_step(
                            step_type=StepType.OBSERVATION,
                            name=getattr(part, "tool_name", "tool_return"),
                            output=str(getattr(part, "content", "")),
                            metadata={
                                "tool_call_id": getattr(
                                    part, "tool_call_id", ""
                                ),
                            },
                        )

            elif msg_kind == "response":
                # Model response — contains parts (text, tool-call)
                parts = getattr(msg, "parts", [])
                for part in parts:
                    part_kind = getattr(part, "part_kind", "")
                    if part_kind == "tool-call":
                        tool_name = getattr(part, "tool_name", "unknown_tool")
                        recorder.add_step(
                            step_type=StepType.TOOL_CALL,
                            name=tool_name,
                            parameters=getattr(part, "args", {}),
                            metadata={
                                "tool_call_id": getattr(
                                    part, "tool_call_id", ""
                                ),
                            },
                        )
                    elif part_kind == "text":
                        recorder.add_step(
                            step_type=StepType.LLM_CALL,
                            name="model_response",
                            output=str(getattr(part, "content", "")),
                        )

    def get_agent_callable(self) -> Callable[[AgentInput], AgentOutput]:
        """Get the adapter as a callable."""
        return self


def wrap_pydantic_ai_agent(
    agent: Any,
    deps: Any = None,
) -> Callable[[AgentInput], AgentOutput]:
    """Wrap a Pydantic AI agent as an Agentrial-compatible agent.

    Args:
        agent: A pydantic_ai.Agent instance.
        deps: Optional dependencies to pass to agent.run_sync().

    Returns:
        Callable that takes AgentInput and returns AgentOutput.

    Example:
        from pydantic_ai import Agent
        agent = Agent('openai:gpt-4o', system_prompt='You are helpful.')
        agent_fn = wrap_pydantic_ai_agent(agent)
    """
    return PydanticAIAdapter(agent, deps)
