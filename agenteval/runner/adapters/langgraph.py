"""LangGraph adapter for AgentEval.

This adapter wraps LangGraph graphs and captures their execution
trajectory using OpenTelemetry spans.
"""

import logging
import time
from typing import Any, Callable

from agenteval.runner.trajectory import TrajectoryRecorder
from agenteval.types import AgentInput, AgentMetadata, AgentOutput, StepType

logger = logging.getLogger(__name__)


class LangGraphAdapter:
    """Adapter for LangGraph graphs.

    This adapter wraps a LangGraph CompiledGraph and converts its
    execution into the AgentEval trajectory format.
    """

    def __init__(
        self,
        graph: Any,
        input_key: str = "messages",
        output_key: str = "messages",
    ) -> None:
        """Initialize the adapter.

        Args:
            graph: A LangGraph CompiledGraph instance.
            input_key: Key in the graph state for input.
            output_key: Key in the graph state for output.
        """
        self.graph = graph
        self.input_key = input_key
        self.output_key = output_key

    def __call__(self, input: AgentInput) -> AgentOutput:
        """Execute the LangGraph and capture trajectory.

        Args:
            input: The agent input.

        Returns:
            AgentOutput with captured trajectory.
        """
        recorder = TrajectoryRecorder()
        start_time = time.time()
        total_tokens = 0
        total_cost = 0.0

        try:
            # Build input state
            # For message-based graphs, wrap query in a HumanMessage
            try:
                from langchain_core.messages import HumanMessage

                input_state = {
                    self.input_key: [HumanMessage(content=input.query)],
                    **input.context,
                }
            except ImportError:
                # langchain not installed, use simple dict
                input_state = {
                    self.input_key: input.query,
                    **input.context,
                }

            # Stream through the graph to capture each step
            final_output = ""

            for event in self.graph.stream(input_state, stream_mode="updates"):
                for node_name, node_output in event.items():
                    # Record this step
                    step_type = self._determine_step_type(node_name, node_output)

                    # Extract relevant information from the node output
                    step_output = self._extract_output(node_output)
                    tokens = self._extract_tokens(node_output)
                    cost = self._extract_cost(node_output)

                    recorder.add_step(
                        step_type=step_type,
                        name=node_name,
                        parameters={"input": str(input_state)[:500]},  # Truncate
                        output=step_output,
                        tokens=tokens,
                        metadata={"raw_output": node_output},
                    )

                    total_tokens += tokens
                    total_cost += cost

                    # Update final output if this looks like the final message
                    if self.output_key in (node_output or {}):
                        messages = node_output[self.output_key]
                        if messages and hasattr(messages[-1], "content"):
                            final_output = messages[-1].content

            # If no output was captured, try to get final state
            if not final_output:
                try:
                    final_state = self.graph.invoke(input_state)
                    if self.output_key in final_state:
                        messages = final_state[self.output_key]
                        if messages:
                            if hasattr(messages[-1], "content"):
                                final_output = messages[-1].content
                            else:
                                final_output = str(messages[-1])
                except Exception:
                    pass

            duration_ms = (time.time() - start_time) * 1000

            return AgentOutput(
                output=final_output,
                steps=recorder.steps,
                metadata=AgentMetadata(
                    total_tokens=total_tokens,
                    cost=total_cost,
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
                metadata=AgentMetadata(
                    total_tokens=total_tokens,
                    cost=total_cost,
                    duration_ms=duration_ms,
                ),
                success=False,
                error=str(e),
            )

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
            # Look for messages
            if "messages" in node_output:
                messages = node_output["messages"]
                if messages and hasattr(messages[-1], "content"):
                    return messages[-1].content
            # Return a summary of keys
            return {k: type(v).__name__ for k, v in node_output.items()}

        if hasattr(node_output, "content"):
            return node_output.content

        return str(node_output)[:500]  # Truncate long outputs

    def _extract_tokens(self, node_output: Any) -> int:
        """Extract token count from node output."""
        if isinstance(node_output, dict):
            # Look for token info in various places
            if "usage" in node_output:
                usage = node_output["usage"]
                return usage.get("total_tokens", 0)
            if "messages" in node_output:
                messages = node_output["messages"]
                if messages:
                    last = messages[-1]
                    if hasattr(last, "usage_metadata"):
                        meta = last.usage_metadata
                        return getattr(meta, "total_tokens", 0)
        return 0

    def _extract_cost(self, node_output: Any) -> float:
        """Extract cost from node output."""
        # Cost is typically not directly available; would need to be calculated
        # based on model and token count
        return 0.0

    def get_agent_callable(self) -> Callable[[AgentInput], AgentOutput]:
        """Get the adapter as a callable."""
        return self


def wrap_langgraph_agent(
    graph: Any,
    input_key: str = "messages",
    output_key: str = "messages",
) -> Callable[[AgentInput], AgentOutput]:
    """Wrap a LangGraph CompiledGraph for use with AgentEval.

    Args:
        graph: A LangGraph CompiledGraph instance.
        input_key: Key in the graph state for input.
        output_key: Key in the graph state for output.

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

        # Use in test suite
        suite = Suite(name="my-suite", agent=agent, ...)
    """
    return LangGraphAdapter(graph, input_key, output_key)
