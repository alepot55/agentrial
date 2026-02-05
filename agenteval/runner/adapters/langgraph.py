"""LangGraph adapter for AgentEval.

This adapter wraps LangGraph graphs and captures their execution
trajectory using OpenTelemetry spans as the primary integration point.
"""

import logging
import time
from typing import Any, Callable

from agenteval.runner.otel import OTelTrajectoryCapture
from agenteval.runner.trajectory import TrajectoryRecorder
from agenteval.types import AgentInput, AgentMetadata, AgentOutput, StepType

logger = logging.getLogger(__name__)


class LangGraphAdapter:
    """Adapter for LangGraph graphs.

    This adapter wraps a LangGraph CompiledGraph and converts its
    execution into the AgentEval trajectory format. It supports two modes:

    1. OTel mode (preferred): Captures spans emitted by LangGraph's built-in
       OpenTelemetry instrumentation. This is the recommended approach as it
       provides richer data and works consistently across LangGraph versions.

    2. Fallback mode: If OTel spans are not available, falls back to extracting
       trajectory data from graph.stream() events.

    The adapter automatically detects which mode to use based on whether
    OTel spans are captured during execution.
    """

    def __init__(
        self,
        graph: Any,
        input_key: str = "messages",
        output_key: str = "messages",
        use_otel: bool = True,
    ) -> None:
        """Initialize the adapter.

        Args:
            graph: A LangGraph CompiledGraph instance.
            input_key: Key in the graph state for input.
            output_key: Key in the graph state for output.
            use_otel: Whether to attempt OTel span capture (default True).
        """
        self.graph = graph
        self.input_key = input_key
        self.output_key = output_key
        self.use_otel = use_otel

    def __call__(self, input: AgentInput) -> AgentOutput:
        """Execute the LangGraph and capture trajectory.

        Args:
            input: The agent input.

        Returns:
            AgentOutput with captured trajectory.
        """
        if self.use_otel:
            return self._execute_with_otel(input)
        return self._execute_with_stream(input)

    def _execute_with_otel(self, input: AgentInput) -> AgentOutput:
        """Execute graph with OTel span capture.

        This is the preferred method. It sets up OTel span capture,
        runs the graph, and collects trajectory from captured spans.
        """
        start_time = time.time()

        with OTelTrajectoryCapture() as capture:
            try:
                input_state = self._build_input_state(input)

                # Invoke the graph
                final_state = self.graph.invoke(input_state)

                # Extract final output
                final_output = self._extract_final_output(final_state)

                duration_ms = (time.time() - start_time) * 1000

                # Get steps from OTel capture
                steps = capture.get_steps()

                # If no OTel spans were captured, fall back to stream mode
                if not steps:
                    logger.debug("No OTel spans captured, falling back to stream mode")
                    return self._execute_with_stream(input)

                # Calculate totals from steps
                total_tokens = sum(s.tokens for s in steps)

                return AgentOutput(
                    output=final_output,
                    steps=steps,
                    metadata=AgentMetadata(
                        total_tokens=total_tokens,
                        cost=self._estimate_cost(total_tokens),
                        duration_ms=duration_ms,
                    ),
                    success=True,
                )

            except Exception as e:
                logger.error("LangGraph execution failed: %s", e)
                duration_ms = (time.time() - start_time) * 1000
                steps = capture.get_steps()

                return AgentOutput(
                    output="",
                    steps=steps,
                    metadata=AgentMetadata(duration_ms=duration_ms),
                    success=False,
                    error=str(e),
                )

    def _execute_with_stream(self, input: AgentInput) -> AgentOutput:
        """Execute graph with stream-based trajectory capture.

        Fallback method when OTel spans are not available.
        """
        recorder = TrajectoryRecorder()
        start_time = time.time()
        total_tokens = 0

        try:
            input_state = self._build_input_state(input)
            final_output = ""

            for event in self.graph.stream(input_state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    step_type = self._determine_step_type(node_name, node_output)
                    step_output = self._extract_output(node_output)
                    tokens = self._extract_tokens(node_output)

                    recorder.add_step(
                        step_type=step_type,
                        name=node_name,
                        parameters={"input": str(input_state)[:500]},
                        output=step_output,
                        tokens=tokens,
                        metadata={"raw_output": node_output},
                    )

                    total_tokens += tokens

                    if self.output_key in (node_output or {}):
                        messages = node_output[self.output_key]
                        if messages and hasattr(messages[-1], "content"):
                            final_output = messages[-1].content

            if not final_output:
                final_state = self.graph.invoke(input_state)
                final_output = self._extract_final_output(final_state)

            duration_ms = (time.time() - start_time) * 1000

            return AgentOutput(
                output=final_output,
                steps=recorder.steps,
                metadata=AgentMetadata(
                    total_tokens=total_tokens,
                    cost=self._estimate_cost(total_tokens),
                    duration_ms=duration_ms,
                ),
                success=True,
            )

        except Exception as e:
            logger.error("LangGraph execution failed: %s", e)
            duration_ms = (time.time() - start_time) * 1000

            return AgentOutput(
                output="",
                steps=recorder.steps,
                metadata=AgentMetadata(duration_ms=duration_ms),
                success=False,
                error=str(e),
            )

    def _build_input_state(self, input: AgentInput) -> dict[str, Any]:
        """Build the input state for the graph."""
        try:
            from langchain_core.messages import HumanMessage

            return {
                self.input_key: [HumanMessage(content=input.query)],
                **input.context,
            }
        except ImportError:
            return {
                self.input_key: input.query,
                **input.context,
            }

    def _extract_final_output(self, state: dict[str, Any]) -> str:
        """Extract final output from graph state."""
        if self.output_key not in state:
            return ""

        messages = state[self.output_key]
        if not messages:
            return ""

        last_message = messages[-1]
        if hasattr(last_message, "content"):
            return last_message.content
        return str(last_message)

    def _determine_step_type(self, node_name: str, output: Any) -> StepType:
        """Determine the step type from node name and output."""
        name_lower = node_name.lower()

        if "tool" in name_lower or "action" in name_lower:
            return StepType.TOOL_CALL
        elif "llm" in name_lower or "model" in name_lower or "agent" in name_lower:
            return StepType.LLM_CALL
        elif "observe" in name_lower or "result" in name_lower:
            return StepType.OBSERVATION
        elif "think" in name_lower or "reason" in name_lower:
            return StepType.REASONING
        else:
            return StepType.OBSERVATION

    def _extract_output(self, node_output: Any) -> Any:
        """Extract readable output from node output."""
        if node_output is None:
            return None

        if isinstance(node_output, dict):
            if "messages" in node_output:
                messages = node_output["messages"]
                if messages and hasattr(messages[-1], "content"):
                    return messages[-1].content
            return {k: type(v).__name__ for k, v in node_output.items()}

        if hasattr(node_output, "content"):
            return node_output.content

        return str(node_output)[:500]

    def _extract_tokens(self, node_output: Any) -> int:
        """Extract token count from node output."""
        if isinstance(node_output, dict):
            if "usage" in node_output:
                return node_output["usage"].get("total_tokens", 0)
            if "messages" in node_output:
                messages = node_output["messages"]
                if messages:
                    last = messages[-1]
                    if hasattr(last, "usage_metadata"):
                        return getattr(last.usage_metadata, "total_tokens", 0)
        return 0

    def _estimate_cost(self, tokens: int) -> float:
        """Estimate cost based on token count.

        Uses GPT-4 pricing as a rough estimate. Actual cost depends on model.
        """
        # GPT-4 ~$0.03/1K input, $0.06/1K output, assume 50/50 split
        return tokens * 0.000045

    def get_agent_callable(self) -> Callable[[AgentInput], AgentOutput]:
        """Get the adapter as a callable."""
        return self


def wrap_langgraph_agent(
    graph: Any,
    input_key: str = "messages",
    output_key: str = "messages",
    use_otel: bool = True,
) -> Callable[[AgentInput], AgentOutput]:
    """Wrap a LangGraph CompiledGraph for use with AgentEval.

    This function creates an adapter that captures the execution trajectory
    of a LangGraph graph. By default, it uses OpenTelemetry spans for
    trajectory capture, which provides the richest data. If OTel spans
    are not available, it falls back to extracting data from stream events.

    Args:
        graph: A LangGraph CompiledGraph instance.
        input_key: Key in the graph state for input (default "messages").
        output_key: Key in the graph state for output (default "messages").
        use_otel: Whether to use OTel span capture (default True).

    Returns:
        A callable that takes AgentInput and returns AgentOutput.

    Example:
        from langgraph.graph import StateGraph
        from agenteval.adapters.langgraph import wrap_langgraph_agent

        # Define your LangGraph
        graph = StateGraph(...)
        compiled = graph.compile()

        # Wrap for AgentEval
        agent = wrap_langgraph_agent(compiled)

        # Define test suite
        suite = Suite(
            name="my-agent-tests",
            agent=agent,
            trials=10,
        )
    """
    return LangGraphAdapter(graph, input_key, output_key, use_otel)
