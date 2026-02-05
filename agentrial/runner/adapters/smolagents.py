"""smolagents (Hugging Face) adapter for Agentrial.

Wraps smolagents agents to capture trajectory and produce
standardized AgentOutput. smolagents is imported only at runtime
so the core package never depends on it.

Usage:
    from agentrial.runner.adapters.smolagents import wrap_smolagent
    agent_fn = wrap_smolagent(my_agent)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from agentrial.runner.adapters.base import BaseAdapter
from agentrial.runner.trajectory import TrajectoryRecorder
from agentrial.types import AgentInput, AgentOutput, StepType


def _import_smolagents() -> Any:
    """Import smolagents at runtime."""
    try:
        import smolagents  # noqa: F811

        return smolagents
    except ImportError as err:
        raise ImportError(
            "smolagents is not installed. Install it with: pip install smolagents"
        ) from err


class SmolAgentsAdapter(BaseAdapter):
    """Adapter for Hugging Face smolagents.

    Wraps a smolagents agent (ToolCallingAgent, CodeAgent, etc.) and
    captures tool calls and LLM interactions from the agent's logs.
    """

    def __init__(
        self,
        agent: Any,
    ) -> None:
        """Initialize the adapter.

        Args:
            agent: A smolagents agent instance.
        """
        self.agent = agent

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

            result = self.agent.run(input.query)
            duration = (time.time() - start) * 1000

            output_text = str(result)

            # Extract trajectory from agent's internal logs
            self._extract_trajectory(recorder)

            agent_output = recorder.finalize(output_text)
            agent_output.metadata.duration_ms = duration
            return agent_output

        except Exception as e:
            return recorder.finalize(
                output="",
                success=False,
                error=f"smolagents execution failed: {e}",
            )

    def _extract_trajectory(self, recorder: TrajectoryRecorder) -> None:
        """Extract trajectory from agent's logs."""
        # smolagents stores execution logs in agent.logs
        logs = getattr(self.agent, "logs", [])

        for log_entry in logs:
            # Each log entry can contain LLM output and tool calls
            if isinstance(log_entry, dict):
                self._process_dict_log(log_entry, recorder)
            else:
                self._process_object_log(log_entry, recorder)

        # Extract token usage from agent's monitor
        monitor = getattr(self.agent, "monitor", None)
        if monitor:
            total_input = getattr(monitor, "total_input_token_count", 0)
            total_output = getattr(monitor, "total_output_token_count", 0)
            if total_input or total_output:
                recorder.add_token_usage(
                    prompt_tokens=total_input,
                    completion_tokens=total_output,
                )

    def _process_dict_log(
        self, entry: dict[str, Any], recorder: TrajectoryRecorder
    ) -> None:
        """Process a dictionary-format log entry."""
        # LLM output
        if "llm_output" in entry:
            recorder.add_step(
                step_type=StepType.LLM_CALL,
                name="llm_response",
                output=str(entry["llm_output"]),
            )

        # Tool calls
        if "tool_call" in entry:
            tool_call = entry["tool_call"]
            recorder.add_step(
                step_type=StepType.TOOL_CALL,
                name=tool_call.get("tool_name", "unknown"),
                parameters=tool_call.get("tool_arguments", {}),
            )

        # Observations (tool outputs)
        if "observation" in entry:
            recorder.add_step(
                step_type=StepType.OBSERVATION,
                name="observation",
                output=str(entry["observation"]),
            )

    def _process_object_log(
        self, entry: Any, recorder: TrajectoryRecorder
    ) -> None:
        """Process an object-format log entry (smolagents step types)."""
        # ActionStep contains tool calls and observations
        if hasattr(entry, "tool_calls") and entry.tool_calls:
            for tc in entry.tool_calls:
                recorder.add_step(
                    step_type=StepType.TOOL_CALL,
                    name=getattr(tc, "name", "unknown"),
                    parameters=getattr(tc, "arguments", {}),
                )

        # LLM output from the step
        if hasattr(entry, "llm_output") and entry.llm_output:
            recorder.add_step(
                step_type=StepType.LLM_CALL,
                name="llm_response",
                output=str(entry.llm_output),
            )

        # Observations
        if hasattr(entry, "observations") and entry.observations:
            recorder.add_step(
                step_type=StepType.OBSERVATION,
                name="observation",
                output=str(entry.observations),
            )

    def get_agent_callable(self) -> Callable[[AgentInput], AgentOutput]:
        """Get the adapter as a callable."""
        return self


def wrap_smolagent(
    agent: Any,
) -> Callable[[AgentInput], AgentOutput]:
    """Wrap a smolagents agent as an Agentrial-compatible agent.

    Args:
        agent: A smolagents agent (ToolCallingAgent, CodeAgent, etc.).

    Returns:
        Callable that takes AgentInput and returns AgentOutput.

    Example:
        from smolagents import ToolCallingAgent, HfApiModel
        agent = ToolCallingAgent(tools=[...], model=HfApiModel())
        agent_fn = wrap_smolagent(agent)
    """
    return SmolAgentsAdapter(agent)
