"""Base adapter interface for framework integrations."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from agentrial.types import AgentInput, AgentOutput


class BaseAdapter(ABC):
    """Abstract base class for framework adapters.

    Adapters wrap framework-specific agents to provide a uniform interface
    for AgentEval. Each adapter is responsible for:
    1. Invoking the wrapped agent
    2. Capturing trajectory steps (tool calls, LLM calls, etc.)
    3. Extracting token usage and cost information
    4. Converting the output to AgentOutput format
    """

    @abstractmethod
    def __call__(self, input: AgentInput) -> AgentOutput:
        """Execute the wrapped agent.

        Args:
            input: The agent input.

        Returns:
            Standardized AgentOutput with trajectory.
        """
        pass

    @abstractmethod
    def get_agent_callable(self) -> Callable[[AgentInput], AgentOutput]:
        """Get the adapter as a callable.

        Returns:
            A callable that takes AgentInput and returns AgentOutput.
        """
        pass


class FunctionAdapter(BaseAdapter):
    """Adapter for simple Python functions.

    This adapter wraps a function that already returns AgentOutput or
    a compatible dict structure.
    """

    def __init__(
        self,
        func: Callable[[AgentInput], AgentOutput | dict[str, Any]],
        name: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            func: The function to wrap.
            name: Optional name for the agent.
        """
        self.func = func
        self.name = name or func.__name__

    def __call__(self, input: AgentInput) -> AgentOutput:
        """Execute the wrapped function.

        Args:
            input: The agent input.

        Returns:
            AgentOutput from the function.
        """
        result = self.func(input)

        if isinstance(result, AgentOutput):
            return result

        # Convert dict to AgentOutput
        if isinstance(result, dict):
            from agentrial.types import AgentMetadata

            return AgentOutput(
                output=result.get("output", ""),
                steps=result.get("steps", []),
                metadata=AgentMetadata(
                    total_tokens=result.get("metadata", {}).get("tokens", 0),
                    cost=result.get("metadata", {}).get("cost", 0.0),
                    duration_ms=result.get("metadata", {}).get("duration", 0.0),
                ),
                success=result.get("success", True),
                error=result.get("error"),
            )

        # Assume result is just the output string
        return AgentOutput(output=str(result))

    def get_agent_callable(self) -> Callable[[AgentInput], AgentOutput]:
        """Get the adapter as a callable."""
        return self


def wrap_function(
    func: Callable[[AgentInput], AgentOutput | dict[str, Any] | str],
    name: str | None = None,
) -> Callable[[AgentInput], AgentOutput]:
    """Wrap a simple function as an AgentEval-compatible agent.

    Args:
        func: The function to wrap.
        name: Optional name for the agent.

    Returns:
        A callable that produces AgentOutput.
    """
    return FunctionAdapter(func, name)
