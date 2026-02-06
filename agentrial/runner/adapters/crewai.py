"""CrewAI adapter for Agentrial.

Wraps CrewAI crews and agents to capture trajectory and produce
standardized AgentOutput. CrewAI is imported only at runtime so the
core package never depends on it.

Usage:
    from agentrial.runner.adapters.crewai import wrap_crewai_agent
    agent_fn = wrap_crewai_agent(my_crew)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from agentrial.runner.adapters.base import BaseAdapter
from agentrial.runner.trajectory import TrajectoryRecorder
from agentrial.types import AgentInput, AgentOutput, StepType


def _import_crewai() -> Any:
    """Import crewai at runtime."""
    try:
        import crewai  # noqa: F811

        return crewai
    except ImportError as err:
        raise ImportError(
            "CrewAI is not installed. Install it with: pip install crewai"
        ) from err


class CrewAIAdapter(BaseAdapter):
    """Adapter for CrewAI crews.

    Supports both Crew.kickoff() and individual Agent execution.
    Captures task completions, tool calls, and token usage from
    CrewAI's internal tracking.
    """

    def __init__(
        self,
        crew: Any,
        input_key: str = "query",
    ) -> None:
        """Initialize the adapter.

        Args:
            crew: A CrewAI Crew instance.
            input_key: Key name for the input passed to kickoff().
        """
        self.crew = crew
        self.input_key = input_key

    def __call__(self, input: AgentInput) -> AgentOutput:
        """Execute the crew and capture trajectory.

        Args:
            input: Agent input with query and context.

        Returns:
            Standardized AgentOutput.
        """
        _import_crewai()
        recorder = TrajectoryRecorder()

        inputs = {self.input_key: input.query, **input.context}

        try:
            start = time.time()
            result = self.crew.kickoff(inputs=inputs)
            duration = (time.time() - start) * 1000

            # Extract trajectory from CrewAI result
            output_text = str(result)

            # CrewAI >= 0.28 provides task_outputs on the result
            if hasattr(result, "tasks_output"):
                for task_output in result.tasks_output:
                    agent_name = "unknown"
                    if hasattr(task_output, "agent") and task_output.agent:
                        agent_name = str(task_output.agent)

                    recorder.add_step(
                        step_type=StepType.TOOL_CALL,
                        name=f"task_{agent_name}",
                        parameters={"task": getattr(task_output, "description", "")},
                        output=str(task_output),
                        metadata={"agent_id": agent_name},
                    )

            # Extract token usage if available
            if hasattr(result, "token_usage"):
                usage = result.token_usage
                recorder.add_token_usage(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    cost=getattr(usage, "total_cost", 0.0),
                )

            agent_output = recorder.finalize(output_text)
            # Override duration with actual measured time
            agent_output.metadata.duration_ms = duration
            return agent_output

        except Exception as e:
            return recorder.finalize(
                output="",
                success=False,
                error=f"CrewAI execution failed: {e}",
            )

    def get_agent_callable(self) -> Callable[[AgentInput], AgentOutput]:
        """Get the adapter as a callable."""
        return self


def wrap_crewai_agent(
    crew: Any,
    input_key: str = "query",
) -> Callable[[AgentInput], AgentOutput]:
    """Wrap a CrewAI crew as an Agentrial-compatible agent.

    Args:
        crew: A CrewAI Crew instance.
        input_key: Key name for input passed to kickoff().

    Returns:
        Callable that takes AgentInput and returns AgentOutput.

    Example:
        from crewai import Crew, Agent, Task
        crew = Crew(agents=[...], tasks=[...])
        agent_fn = wrap_crewai_agent(crew)
    """
    return CrewAIAdapter(crew, input_key)
