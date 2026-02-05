"""OpenTelemetry span processing for Agentrial.

This module provides the universal integration point for all framework adapters.
Frameworks that emit OTel spans (LangGraph, Pydantic AI, OpenAI SDK, etc.) can
be captured through this processor.
"""

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from agentrial.types import StepType, TrajectoryStep

logger = logging.getLogger(__name__)


# Mapping of common span names/attributes to step types
SPAN_TYPE_MAPPINGS = {
    # LangGraph
    "langgraph.node": StepType.LLM_CALL,
    "langgraph.tool": StepType.TOOL_CALL,
    # LangChain
    "langchain.llm": StepType.LLM_CALL,
    "langchain.chain": StepType.LLM_CALL,
    "langchain.tool": StepType.TOOL_CALL,
    "langchain.agent": StepType.LLM_CALL,
    # OpenAI
    "openai.chat": StepType.LLM_CALL,
    "openai.completion": StepType.LLM_CALL,
    # Generic patterns
    "llm": StepType.LLM_CALL,
    "tool": StepType.TOOL_CALL,
    "agent": StepType.LLM_CALL,
}


def _determine_step_type(span: ReadableSpan) -> StepType:
    """Determine step type from span name and attributes."""
    name = span.name.lower()

    # Check explicit mappings
    for pattern, step_type in SPAN_TYPE_MAPPINGS.items():
        if pattern in name:
            return step_type

    # Check attributes
    attributes = dict(span.attributes or {})
    if "tool.name" in attributes or "tool_name" in attributes:
        return StepType.TOOL_CALL
    if "llm.model" in attributes or "model" in attributes:
        return StepType.LLM_CALL

    # Default based on name heuristics
    if "tool" in name or "action" in name or "execute" in name:
        return StepType.TOOL_CALL
    if "llm" in name or "model" in name or "chat" in name or "complete" in name:
        return StepType.LLM_CALL
    if "think" in name or "reason" in name:
        return StepType.REASONING
    if "observe" in name or "result" in name:
        return StepType.OBSERVATION

    return StepType.OBSERVATION


def _extract_tokens(span: ReadableSpan) -> int:
    """Extract token count from span attributes."""
    attributes = dict(span.attributes or {})

    # Standard OTel semantic conventions for LLM
    if "llm.usage.total_tokens" in attributes:
        return int(attributes["llm.usage.total_tokens"])
    if "gen_ai.usage.total_tokens" in attributes:
        return int(attributes["gen_ai.usage.total_tokens"])

    # LangChain/LangGraph conventions
    if "total_tokens" in attributes:
        return int(attributes["total_tokens"])

    # Sum prompt + completion if available
    prompt = attributes.get("llm.usage.prompt_tokens", 0)
    completion = attributes.get("llm.usage.completion_tokens", 0)
    if prompt or completion:
        return int(prompt) + int(completion)

    return 0


def _extract_parameters(span: ReadableSpan) -> dict[str, Any]:
    """Extract relevant parameters from span attributes."""
    attributes = dict(span.attributes or {})
    params = {}

    # Extract input/query
    for key in ["input", "query", "prompt", "messages", "llm.input"]:
        if key in attributes:
            value = attributes[key]
            # Truncate long values
            if isinstance(value, str) and len(value) > 500:
                value = value[:500] + "..."
            params[key] = value

    # Extract tool parameters
    for key in ["tool.parameters", "tool_input", "args"]:
        if key in attributes:
            params["tool_params"] = attributes[key]

    return params


def _extract_output(span: ReadableSpan) -> Any:
    """Extract output from span attributes."""
    attributes = dict(span.attributes or {})

    # Check common output keys
    for key in ["output", "result", "response", "llm.output", "tool.output"]:
        if key in attributes:
            value = attributes[key]
            if isinstance(value, str) and len(value) > 500:
                return value[:500] + "..."
            return value

    return None


class TrajectorySpanExporter(SpanExporter):
    """Span exporter that collects spans into trajectory steps.

    This exporter is registered with the OTel SDK and receives all spans
    emitted during agent execution.
    """

    def __init__(self) -> None:
        """Initialize the exporter."""
        self._steps: list[TrajectoryStep] = []
        self._step_index = 0

    def export(self, spans: list[ReadableSpan]) -> SpanExportResult:
        """Export spans by converting them to trajectory steps.

        Args:
            spans: Spans to export.

        Returns:
            SpanExportResult.SUCCESS
        """
        for span in spans:
            # Skip internal/infrastructure spans
            if self._should_skip_span(span):
                continue

            step = TrajectoryStep(
                step_index=self._step_index,
                step_type=_determine_step_type(span),
                name=span.name,
                parameters=_extract_parameters(span),
                output=_extract_output(span),
                duration_ms=(span.end_time - span.start_time) / 1_000_000
                if span.end_time and span.start_time
                else 0,
                tokens=_extract_tokens(span),
                metadata={
                    "trace_id": format(span.context.trace_id, "032x") if span.context else None,
                    "span_id": format(span.context.span_id, "016x") if span.context else None,
                    "attributes": dict(span.attributes or {}),
                },
            )
            self._steps.append(step)
            self._step_index += 1

        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush pending spans."""
        return True

    def _should_skip_span(self, span: ReadableSpan) -> bool:
        """Determine if a span should be skipped."""
        name = span.name.lower()
        # Skip HTTP/network infrastructure spans
        if any(prefix in name for prefix in ["http.", "grpc.", "dns.", "tcp."]):
            return True
        # Skip very short spans (likely infrastructure)
        if span.end_time and span.start_time:
            duration_ms = (span.end_time - span.start_time) / 1_000_000
            if duration_ms < 1:  # Less than 1ms
                return True
        return False

    def get_steps(self) -> list[TrajectoryStep]:
        """Get collected trajectory steps."""
        return self._steps.copy()

    def clear(self) -> None:
        """Clear collected steps."""
        self._steps.clear()
        self._step_index = 0


class OTelTrajectoryCapture:
    """Context manager for capturing OTel spans during agent execution.

    This provides a clean API for capturing trajectories from any
    OTel-instrumented framework.

    Example:
        with OTelTrajectoryCapture() as capture:
            result = my_agent(input)
        steps = capture.get_steps()
    """

    def __init__(self) -> None:
        """Initialize the capture."""
        self._exporter: TrajectorySpanExporter | None = None
        self._provider: TracerProvider | None = None
        self._previous_provider: Any = None

    def __enter__(self) -> "OTelTrajectoryCapture":
        """Start capturing spans."""
        self._exporter = TrajectorySpanExporter()
        self._provider = TracerProvider()
        self._provider.add_span_processor(SimpleSpanProcessor(self._exporter))

        # Store previous provider and set ours
        self._previous_provider = trace.get_tracer_provider()
        trace.set_tracer_provider(self._provider)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Stop capturing and restore previous provider."""
        if self._provider:
            self._provider.shutdown()

        # Restore previous provider
        if self._previous_provider:
            trace.set_tracer_provider(self._previous_provider)

    def get_steps(self) -> list[TrajectoryStep]:
        """Get captured trajectory steps."""
        if self._exporter:
            return self._exporter.get_steps()
        return []

    def clear(self) -> None:
        """Clear captured steps."""
        if self._exporter:
            self._exporter.clear()
