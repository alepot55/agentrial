"""AutoGen adapter for Agentrial.

Wraps AutoGen agents (v0.4+) to capture trajectory and produce
standardized AgentOutput. AutoGen is imported only at runtime so the
core package never depends on it.

Usage:
    from agentrial.runner.adapters.autogen import wrap_autogen_agent
    agent_fn = wrap_autogen_agent(my_agent, task_fn=my_task)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from agentrial.runner.adapters.base import BaseAdapter
from agentrial.runner.trajectory import TrajectoryRecorder
from agentrial.types import AgentInput, AgentOutput, StepType


def _import_autogen() -> Any:
    """Import autogen at runtime."""
    try:
        import autogen_agentchat  # noqa: F811

        return autogen_agentchat
    except ImportError:
        pass

    try:
        import autogen  # noqa: F811

        return autogen
    except ImportError as err:
        raise ImportError(
            "AutoGen is not installed. Install it with: "
            "pip install autogen-agentchat  (v0.4+) or pip install pyautogen"
        ) from err


class AutoGenAdapter(BaseAdapter):
    """Adapter for AutoGen agents.

    Supports both AutoGen v0.4+ (autogen-agentchat) and legacy pyautogen.
    For v0.4+, wraps the async runtime. For legacy, wraps initiate_chat.
    """

    def __init__(
        self,
        agent: Any,
        task_fn: Callable[[str], Any] | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            agent: An AutoGen agent or team instance.
            task_fn: Optional function that creates a task from a query string.
                     For v0.4+: should return a TextMessage or similar.
                     For legacy: not needed; initiate_chat is used directly.
        """
        self.agent = agent
        self.task_fn = task_fn

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

            # Detect v0.4+ vs legacy
            if self._is_v04():
                output_text = self._run_v04(input, recorder)
            else:
                output_text = self._run_legacy(input, recorder)

            duration = (time.time() - start) * 1000

            agent_output = recorder.finalize(output_text)
            agent_output.metadata.duration_ms = duration
            return agent_output

        except Exception as e:
            return recorder.finalize(
                output="",
                success=False,
                error=f"AutoGen execution failed: {e}",
            )

    def _is_v04(self) -> bool:
        """Check if we're using AutoGen v0.4+."""
        try:
            import autogen_agentchat  # noqa: F401

            return True
        except ImportError:
            return False

    def _run_v04(self, input: AgentInput, recorder: TrajectoryRecorder) -> str:
        """Run using AutoGen v0.4+ async runtime."""
        import asyncio

        from autogen_agentchat.messages import TextMessage

        async def _execute() -> str:
            if self.task_fn:
                task = self.task_fn(input.query)
            else:
                task = TextMessage(content=input.query, source="user")

            # For team agents (e.g., RoundRobinGroupChat)
            if hasattr(self.agent, "run_stream"):
                messages = []
                async for message in self.agent.run_stream(task=task):
                    if hasattr(message, "content") and hasattr(message, "source"):
                        messages.append(message)
                        recorder.add_step(
                            step_type=StepType.LLM_CALL,
                            name=f"agent_{getattr(message, 'source', 'unknown')}",
                            output=str(message.content),
                            metadata={
                                "agent_id": getattr(message, "source", "unknown"),
                            },
                        )
                return str(messages[-1].content) if messages else ""

            # For single agents
            if hasattr(self.agent, "run"):
                result = await self.agent.run(task=task)
                if hasattr(result, "messages"):
                    for msg in result.messages:
                        if hasattr(msg, "content"):
                            recorder.add_step(
                                step_type=StepType.LLM_CALL,
                                name=f"agent_{getattr(msg, 'source', 'unknown')}",
                                output=str(msg.content),
                                metadata={
                                    "agent_id": getattr(msg, "source", "unknown"),
                                },
                            )
                return str(result)

            return ""

        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return loop.run_in_executor(pool, asyncio.run, _execute())  # type: ignore[arg-type]
        except RuntimeError:
            return asyncio.run(_execute())

    def _run_legacy(self, input: AgentInput, recorder: TrajectoryRecorder) -> str:
        """Run using legacy pyautogen."""
        # For legacy autogen, use initiate_chat pattern
        import autogen

        # Create a user proxy for the conversation
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config=False,
        )

        chat_result = user_proxy.initiate_chat(
            self.agent,
            message=input.query,
        )

        # Extract trajectory from chat history
        if hasattr(chat_result, "chat_history"):
            for msg in chat_result.chat_history:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                name = msg.get("name", role)

                if role == "assistant":
                    step_type = StepType.LLM_CALL
                elif "function_call" in msg or "tool_calls" in msg:
                    step_type = StepType.TOOL_CALL
                else:
                    step_type = StepType.OBSERVATION

                recorder.add_step(
                    step_type=step_type,
                    name=name,
                    output=content,
                    metadata={"agent_id": name, "role": role},
                )

        # Extract cost if available
        if hasattr(chat_result, "cost"):
            cost = chat_result.cost or {}
            total_cost = sum(
                v.get("cost", 0) for v in cost.values() if isinstance(v, dict)
            )
            recorder.add_token_usage(cost=total_cost)

        # Get final output
        if hasattr(chat_result, "summary"):
            return str(chat_result.summary)
        if hasattr(chat_result, "chat_history") and chat_result.chat_history:
            return str(chat_result.chat_history[-1].get("content", ""))
        return ""

    def get_agent_callable(self) -> Callable[[AgentInput], AgentOutput]:
        """Get the adapter as a callable."""
        return self


def wrap_autogen_agent(
    agent: Any,
    task_fn: Callable[[str], Any] | None = None,
) -> Callable[[AgentInput], AgentOutput]:
    """Wrap an AutoGen agent as an Agentrial-compatible agent.

    Args:
        agent: An AutoGen agent or team instance.
        task_fn: Optional function to create tasks from query strings.

    Returns:
        Callable that takes AgentInput and returns AgentOutput.

    Example (v0.4+):
        from autogen_agentchat.agents import AssistantAgent
        agent = AssistantAgent(name="assistant", model_client=client)
        agent_fn = wrap_autogen_agent(agent)

    Example (legacy):
        import autogen
        agent = autogen.AssistantAgent(name="assistant", llm_config=config)
        agent_fn = wrap_autogen_agent(agent)
    """
    return AutoGenAdapter(agent, task_fn)
