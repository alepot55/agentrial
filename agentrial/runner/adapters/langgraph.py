"""LangGraph adapter for Agentrial.

This adapter wraps LangGraph graphs and captures their execution
trajectory using LangChain callback handlers (primary) or OpenTelemetry
spans (optional fallback).
"""

import logging
import time
import warnings
from collections.abc import Callable
from typing import Any
from uuid import UUID

from agentrial.runner.adapters.pricing import calculate_cost
from agentrial.runner.otel import OTelTrajectoryCapture
from agentrial.runner.trajectory import TrajectoryRecorder
from agentrial.types import AgentInput, AgentMetadata, AgentOutput, StepType, TrajectoryStep

# Silence LangGraph deprecation warnings about create_react_agent
warnings.filterwarnings(
    "ignore",
    message=".*create_react_agent has been moved.*",
    category=DeprecationWarning,
)

logger = logging.getLogger(__name__)


class TrajectoryCallbackHandler:
    """LangChain callback handler for capturing trajectory.

    This handler intercepts LangChain/LangGraph events and records
    them as trajectory steps for Agentrial analysis.
    """

    def __init__(self) -> None:
        """Initialize the callback handler."""
        self.steps: list[TrajectoryStep] = []
        self.step_index = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.model_name: str | None = None
        self._pending_tool_calls: dict[str, dict[str, Any]] = {}
        self._step_start_times: dict[str, float] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record tool call start."""
        tool_name = serialized.get("name", "unknown_tool")
        self._step_start_times[str(run_id)] = time.time()

        # Parse inputs
        params = inputs if inputs else {"input": input_str}

        self._pending_tool_calls[str(run_id)] = {
            "step_index": self.step_index,
            "name": tool_name,
            "parameters": params,
            "start_time": time.time(),
        }
        self.step_index += 1

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record tool call end with output."""
        run_id_str = str(run_id)
        if run_id_str in self._pending_tool_calls:
            pending = self._pending_tool_calls.pop(run_id_str)
            duration_ms = (time.time() - pending["start_time"]) * 1000

            # Create tool call step
            self.steps.append(TrajectoryStep(
                step_index=pending["step_index"],
                step_type=StepType.TOOL_CALL,
                name=pending["name"],
                parameters=pending["parameters"],
                output=str(output)[:1000] if output else None,
                duration_ms=duration_ms,
                tokens=0,
            ))

            # Create observation step
            self.steps.append(TrajectoryStep(
                step_index=self.step_index,
                step_type=StepType.OBSERVATION,
                name=f"tool_result_{pending['name']}",
                parameters={},
                output=str(output)[:1000] if output else None,
                duration_ms=0,
                tokens=0,
            ))
            self.step_index += 1

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record tool error."""
        run_id_str = str(run_id)
        if run_id_str in self._pending_tool_calls:
            pending = self._pending_tool_calls.pop(run_id_str)
            duration_ms = (time.time() - pending["start_time"]) * 1000

            self.steps.append(TrajectoryStep(
                step_index=pending["step_index"],
                step_type=StepType.TOOL_CALL,
                name=pending["name"],
                parameters=pending["parameters"],
                output=f"Error: {error}",
                duration_ms=duration_ms,
                tokens=0,
            ))

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Record LLM call start."""
        self._step_start_times[str(run_id)] = time.time()

        # Try to extract model name
        if "kwargs" in serialized:
            self.model_name = (
                serialized["kwargs"].get("model_name")
                or serialized["kwargs"].get("model")
            )
        if not self.model_name and "name" in serialized:
            self.model_name = serialized["name"]

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record LLM call end with response and token usage."""
        run_id_str = str(run_id)
        start_time = self._step_start_times.pop(run_id_str, time.time())
        duration_ms = (time.time() - start_time) * 1000

        # Extract token usage from response
        input_tokens = 0
        output_tokens = 0
        output_text = ""

        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("usage", {})
            input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

        # Try to get usage from generations
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "text"):
                        output_text = gen.text
                    if hasattr(gen, "message"):
                        msg = gen.message
                        if hasattr(msg, "content"):
                            output_text = msg.content
                        # Extract usage_metadata from message (langchain-anthropic style)
                        if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                            um = msg.usage_metadata
                            input_tokens = getattr(um, "input_tokens", 0) or 0
                            output_tokens = getattr(um, "output_tokens", 0) or 0
                        # Also check response_metadata
                        if hasattr(msg, "response_metadata") and msg.response_metadata:
                            rm = msg.response_metadata
                            if "usage" in rm:
                                usage = rm["usage"]
                                input_tokens = usage.get("input_tokens", input_tokens)
                                output_tokens = usage.get("output_tokens", output_tokens)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Check if this was a tool call decision
        has_tool_calls = False
        if hasattr(response, "generations") and response.generations:
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "message") and hasattr(gen.message, "tool_calls"):
                        if gen.message.tool_calls:
                            has_tool_calls = True

        step_type = StepType.TOOL_CALL if has_tool_calls else StepType.LLM_CALL
        step_name = "tool_decision" if has_tool_calls else "llm_response"

        self.steps.append(TrajectoryStep(
            step_index=self.step_index,
            step_type=step_type,
            name=step_name,
            parameters={},
            output=output_text[:1000] if output_text else None,
            duration_ms=duration_ms,
            tokens=input_tokens + output_tokens,
            metadata={"input_tokens": input_tokens, "output_tokens": output_tokens},
        ))
        self.step_index += 1

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        """Record LLM error."""
        run_id_str = str(run_id)
        start_time = self._step_start_times.pop(run_id_str, time.time())
        duration_ms = (time.time() - start_time) * 1000

        self.steps.append(TrajectoryStep(
            step_index=self.step_index,
            step_type=StepType.LLM_CALL,
            name="llm_error",
            parameters={},
            output=f"Error: {error}",
            duration_ms=duration_ms,
            tokens=0,
        ))
        self.step_index += 1

    def get_total_tokens(self) -> int:
        """Get total token count."""
        return self.total_input_tokens + self.total_output_tokens

    def get_cost(self) -> float:
        """Calculate cost based on model and token usage."""
        if self.model_name:
            return calculate_cost(
                self.model_name,
                self.total_input_tokens,
                self.total_output_tokens,
            )
        # Fallback estimate
        return self.get_total_tokens() * 0.000045


class LangGraphAdapter:
    """Adapter for LangGraph graphs.

    This adapter wraps a LangGraph CompiledGraph and converts its
    execution into the Agentrial trajectory format.

    Primary mode: Uses LangChain callback handlers to intercept events.
    Fallback mode: Uses stream events or OTel spans if callbacks unavailable.
    """

    def __init__(
        self,
        graph: Any,
        input_key: str = "messages",
        output_key: str = "messages",
        use_callbacks: bool = True,
        use_otel: bool = False,
    ) -> None:
        """Initialize the adapter.

        Args:
            graph: A LangGraph CompiledGraph instance.
            input_key: Key in the graph state for input.
            output_key: Key in the graph state for output.
            use_callbacks: Whether to use callback handlers (default True).
            use_otel: Whether to use OTel spans as fallback (default False).
        """
        self.graph = graph
        self.input_key = input_key
        self.output_key = output_key
        self.use_callbacks = use_callbacks
        self.use_otel = use_otel

    def __call__(self, input: AgentInput) -> AgentOutput:
        """Execute the LangGraph and capture trajectory.

        Args:
            input: The agent input.

        Returns:
            AgentOutput with captured trajectory.
        """
        if self.use_callbacks:
            return self._execute_with_callbacks(input)
        elif self.use_otel:
            return self._execute_with_otel(input)
        return self._execute_with_stream(input)

    def _execute_with_callbacks(self, input: AgentInput) -> AgentOutput:
        """Execute graph with callback-based trajectory capture.

        This is the primary method using LangChain callbacks.
        """
        start_time = time.time()
        handler = TrajectoryCallbackHandler()

        try:
            input_state = self._build_input_state(input)

            # Try to create a LangChain callback handler wrapper
            try:
                from langchain_core.callbacks import BaseCallbackHandler

                # Create a wrapper that delegates to our handler
                class CallbackWrapper(BaseCallbackHandler):
                    def __init__(self, trajectory_handler: TrajectoryCallbackHandler):
                        self.handler = trajectory_handler

                    def on_tool_start(self, serialized, input_str, **kwargs):
                        self.handler.on_tool_start(serialized, input_str, **kwargs)

                    def on_tool_end(self, output, **kwargs):
                        self.handler.on_tool_end(output, **kwargs)

                    def on_tool_error(self, error, **kwargs):
                        self.handler.on_tool_error(error, **kwargs)

                    def on_llm_start(self, serialized, prompts, **kwargs):
                        self.handler.on_llm_start(serialized, prompts, **kwargs)

                    def on_llm_end(self, response, **kwargs):
                        self.handler.on_llm_end(response, **kwargs)

                    def on_llm_error(self, error, **kwargs):
                        self.handler.on_llm_error(error, **kwargs)

                callback_wrapper = CallbackWrapper(handler)

                # Invoke with callbacks
                config = {"callbacks": [callback_wrapper]}
                final_state = self.graph.invoke(input_state, config=config)

            except ImportError:
                logger.warning("langchain_core not available, falling back to stream mode")
                return self._execute_with_stream(input)

            final_output = self._extract_final_output(final_state)
            duration_ms = (time.time() - start_time) * 1000

            # Sort steps by index
            steps = sorted(handler.steps, key=lambda s: s.step_index)

            return AgentOutput(
                output=final_output,
                steps=steps,
                metadata=AgentMetadata(
                    total_tokens=handler.get_total_tokens(),
                    cost=handler.get_cost(),
                    duration_ms=duration_ms,
                ),
                success=True,
            )

        except Exception as e:
            logger.error("LangGraph execution failed: %s", e)
            duration_ms = (time.time() - start_time) * 1000

            return AgentOutput(
                output="",
                steps=handler.steps,
                metadata=AgentMetadata(
                    total_tokens=handler.get_total_tokens(),
                    cost=handler.get_cost(),
                    duration_ms=duration_ms,
                ),
                success=False,
                error=str(e),
            )

    def _execute_with_otel(self, input: AgentInput) -> AgentOutput:
        """Execute graph with OTel span capture (fallback)."""
        start_time = time.time()

        with OTelTrajectoryCapture() as capture:
            try:
                input_state = self._build_input_state(input)
                final_state = self.graph.invoke(input_state)
                final_output = self._extract_final_output(final_state)
                duration_ms = (time.time() - start_time) * 1000

                steps = capture.get_steps()
                if not steps:
                    logger.debug("No OTel spans captured, falling back to stream mode")
                    return self._execute_with_stream(input)

                total_tokens = sum(s.tokens for s in steps)

                return AgentOutput(
                    output=final_output,
                    steps=steps,
                    metadata=AgentMetadata(
                        total_tokens=total_tokens,
                        cost=total_tokens * 0.000045,
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
        """Execute graph with stream-based trajectory capture (fallback)."""
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
                    cost=total_tokens * 0.000045,
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
                    if hasattr(last, "usage_metadata") and last.usage_metadata:
                        um = last.usage_metadata
                        input_tok = getattr(um, "input_tokens", 0) or 0
                        output_tok = getattr(um, "output_tokens", 0) or 0
                        return input_tok + output_tok
        return 0

    def get_agent_callable(self) -> Callable[[AgentInput], AgentOutput]:
        """Get the adapter as a callable."""
        return self


def wrap_langgraph_agent(
    graph: Any,
    input_key: str = "messages",
    output_key: str = "messages",
    use_callbacks: bool = True,
    use_otel: bool = False,
) -> Callable[[AgentInput], AgentOutput]:
    """Wrap a LangGraph CompiledGraph for use with Agentrial.

    This function creates an adapter that captures the execution trajectory
    of a LangGraph graph using LangChain callback handlers (primary method)
    or OTel spans / stream events (fallback).

    Args:
        graph: A LangGraph CompiledGraph instance.
        input_key: Key in the graph state for input (default "messages").
        output_key: Key in the graph state for output (default "messages").
        use_callbacks: Whether to use callback handlers (default True, recommended).
        use_otel: Whether to use OTel spans as fallback (default False).

    Returns:
        A callable that takes AgentInput and returns AgentOutput.

    Example:
        from langgraph.graph import StateGraph
        from agentrial.runner.adapters import wrap_langgraph_agent

        # Define your LangGraph
        graph = StateGraph(...)
        compiled = graph.compile()

        # Wrap for Agentrial
        agent = wrap_langgraph_agent(compiled)

        # Use in test suite
        # In agentrial.yml, reference the wrapped agent
    """
    return LangGraphAdapter(graph, input_key, output_key, use_callbacks, use_otel)
