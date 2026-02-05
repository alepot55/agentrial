"""Trajectory recording and management."""

import time
from contextlib import contextmanager
from typing import Any, Generator

from agentrial.types import AgentMetadata, AgentOutput, StepType, TrajectoryStep


class TrajectoryRecorder:
    """Records trajectory steps during agent execution.

    This class is used to capture the step-by-step execution of an agent,
    including tool calls, LLM responses, and timing information.

    Usage:
        recorder = TrajectoryRecorder()
        with recorder.record_step(StepType.TOOL_CALL, "search_flights") as step:
            result = call_tool(...)
            step.output = result
            step.tokens = 100
        output = recorder.finalize("Final answer")
    """

    def __init__(self) -> None:
        """Initialize the recorder."""
        self._steps: list[TrajectoryStep] = []
        self._start_time: float = time.time()
        self._total_tokens: int = 0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._cost: float = 0.0

    @contextmanager
    def record_step(
        self,
        step_type: StepType,
        name: str,
        parameters: dict[str, Any] | None = None,
    ) -> Generator[TrajectoryStep, None, None]:
        """Context manager to record a trajectory step.

        Args:
            step_type: Type of step being recorded.
            name: Name of the step (e.g., tool name).
            parameters: Input parameters for this step.

        Yields:
            The TrajectoryStep being recorded. Caller can set output and
            other fields before the context exits.
        """
        step = TrajectoryStep(
            step_index=len(self._steps),
            step_type=step_type,
            name=name,
            parameters=parameters or {},
        )

        start = time.time()
        try:
            yield step
        finally:
            step.duration_ms = (time.time() - start) * 1000
            self._steps.append(step)
            self._total_tokens += step.tokens

    def add_step(
        self,
        step_type: StepType,
        name: str,
        parameters: dict[str, Any] | None = None,
        output: Any = None,
        duration_ms: float = 0.0,
        tokens: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> TrajectoryStep:
        """Add a completed step to the trajectory.

        Args:
            step_type: Type of step.
            name: Name of the step.
            parameters: Input parameters.
            output: Output from the step.
            duration_ms: Duration in milliseconds.
            tokens: Tokens used.
            metadata: Additional metadata.

        Returns:
            The created TrajectoryStep.
        """
        step = TrajectoryStep(
            step_index=len(self._steps),
            step_type=step_type,
            name=name,
            parameters=parameters or {},
            output=output,
            duration_ms=duration_ms,
            tokens=tokens,
            metadata=metadata or {},
        )
        self._steps.append(step)
        self._total_tokens += tokens
        return step

    def add_token_usage(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Add token usage information.

        Args:
            prompt_tokens: Tokens used for prompts.
            completion_tokens: Tokens used for completions.
            cost: Cost in USD.
        """
        self._prompt_tokens += prompt_tokens
        self._completion_tokens += completion_tokens
        self._total_tokens += prompt_tokens + completion_tokens
        self._cost += cost

    def finalize(
        self,
        output: str,
        success: bool = True,
        error: str | None = None,
    ) -> AgentOutput:
        """Finalize recording and return the AgentOutput.

        Args:
            output: The final output string from the agent.
            success: Whether the agent completed successfully.
            error: Error message if the agent failed.

        Returns:
            Complete AgentOutput with trajectory and metadata.
        """
        total_duration = (time.time() - self._start_time) * 1000

        return AgentOutput(
            output=output,
            steps=self._steps.copy(),
            metadata=AgentMetadata(
                total_tokens=self._total_tokens,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                cost=self._cost,
                duration_ms=total_duration,
            ),
            success=success,
            error=error,
        )

    @property
    def steps(self) -> list[TrajectoryStep]:
        """Get the recorded steps."""
        return self._steps.copy()


def extract_trajectory_from_otel_spans(spans: list[dict[str, Any]]) -> list[TrajectoryStep]:
    """Extract trajectory steps from OpenTelemetry spans.

    Args:
        spans: List of OTel span dictionaries.

    Returns:
        List of TrajectoryStep objects.
    """
    steps = []
    for i, span in enumerate(sorted(spans, key=lambda s: s.get("start_time", 0))):
        # Determine step type from span attributes
        span_name = span.get("name", "unknown")
        attributes = span.get("attributes", {})

        if "tool" in span_name.lower() or attributes.get("tool.name"):
            step_type = StepType.TOOL_CALL
            name = attributes.get("tool.name", span_name)
        elif "llm" in span_name.lower() or attributes.get("llm.model"):
            step_type = StepType.LLM_CALL
            name = attributes.get("llm.model", span_name)
        else:
            step_type = StepType.OBSERVATION
            name = span_name

        # Calculate duration
        start = span.get("start_time", 0)
        end = span.get("end_time", start)
        duration_ms = (end - start) / 1_000_000  # nanoseconds to milliseconds

        step = TrajectoryStep(
            step_index=i,
            step_type=step_type,
            name=name,
            parameters=attributes.get("input", {}),
            output=attributes.get("output"),
            duration_ms=duration_ms,
            tokens=attributes.get("tokens", 0),
            metadata=attributes,
        )
        steps.append(step)

    return steps
