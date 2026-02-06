"""Pytest plugin for Agentrial.

Allows running agent evaluations as standard pytest tests using
the @agent_test decorator.

Usage:
    # test_my_agent.py
    from agentrial.pytest_plugin import agent_test

    @agent_test(agent="my_module.my_agent", trials=10, threshold=0.8)
    def test_flight_search(agent_result):
        assert agent_result.tool_called("search_flights")
        assert "cheapest" in agent_result.output
        assert agent_result.cost < 0.15

    # Run with: pytest tests/ --agenteval
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest

from agentrial.types import AgentInput, AgentOutput


@dataclass
class AgentResult:
    """Convenience wrapper around AgentOutput for pytest assertions.

    Provides a user-friendly interface for making assertions about
    agent execution in pytest tests.
    """

    output: str
    steps: list[Any]
    cost: float
    tokens: int
    duration_ms: float
    success: bool
    error: str | None
    pass_rate: float
    trials_passed: int
    trials_total: int
    trial_results: list[dict[str, Any]] = field(default_factory=list)
    _raw_outputs: list[AgentOutput] = field(default_factory=list)

    def tool_called(self, tool_name: str) -> bool:
        """Check if a tool was called in any trial.

        Args:
            tool_name: Name of the tool to check for.

        Returns:
            True if the tool was called in at least one trial.
        """
        for raw in self._raw_outputs:
            for step in raw.steps:
                if step.name == tool_name:
                    return True
        return False

    def tool_called_in_all(self, tool_name: str) -> bool:
        """Check if a tool was called in every trial.

        Args:
            tool_name: Name of the tool to check for.

        Returns:
            True if the tool was called in every trial.
        """
        for raw in self._raw_outputs:
            found = any(step.name == tool_name for step in raw.steps)
            if not found:
                return False
        return len(self._raw_outputs) > 0

    def output_contains(self, text: str) -> bool:
        """Check if the output contains text in any trial.

        Args:
            text: Text to search for.

        Returns:
            True if found in any trial's output.
        """
        return any(text in raw.output for raw in self._raw_outputs)

    def output_contains_in_all(self, text: str) -> bool:
        """Check if output contains text in all trials.

        Args:
            text: Text to search for.

        Returns:
            True if found in every trial's output.
        """
        return all(text in raw.output for raw in self._raw_outputs)

    @property
    def mean_cost(self) -> float:
        """Mean cost across all trials."""
        if not self._raw_outputs:
            return 0.0
        return sum(r.metadata.cost for r in self._raw_outputs) / len(
            self._raw_outputs
        )

    @property
    def mean_tokens(self) -> float:
        """Mean tokens across all trials."""
        if not self._raw_outputs:
            return 0.0
        return sum(r.metadata.total_tokens for r in self._raw_outputs) / len(
            self._raw_outputs
        )


@dataclass
class _AgentTestConfig:
    """Internal config for an agent_test-decorated function."""

    agent: str | Callable[..., Any] | None = None
    query: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    trials: int = 10
    threshold: float = 0.8


def agent_test(
    agent: str | Callable[..., Any] | None = None,
    query: str = "",
    context: dict[str, Any] | None = None,
    trials: int = 10,
    threshold: float = 0.8,
) -> Callable[..., Any]:
    """Decorator for agent evaluation tests.

    The decorated function receives an AgentResult object and should use
    standard pytest assertions to validate the agent's behavior.

    Args:
        agent: Agent callable or import path string. If None, must be
               provided via --agent CLI flag or fixture.
        query: The query to send to the agent.
        context: Additional context for the agent.
        trials: Number of trials to run (default 10).
        threshold: Minimum pass rate to consider passing (default 0.8).

    Returns:
        Decorated test function.

    Example:
        @agent_test(agent="my_module.agent", query="What is 2+2?", trials=5)
        def test_math(agent_result):
            assert "4" in agent_result.output
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        config = _AgentTestConfig(
            agent=agent,
            query=query,
            context=context or {},
            trials=trials,
            threshold=threshold,
        )

        # Store config on the function for the plugin to find
        func._agentrial_config = config  # type: ignore[attr-defined]

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # If called with agent_result fixture, use it
            if "agent_result" in kwargs:
                return func(*args, **kwargs)

            # Otherwise, run the agent and create agent_result
            result = _run_agent_test(config, func)
            return result

        # Override signature so pytest doesn't try to inject
        # the original function's parameters as fixtures
        wrapper.__signature__ = inspect.Signature(parameters=[])
        wrapper._agentrial_config = config  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _run_agent_test(
    config: _AgentTestConfig,
    check_fn: Callable[..., Any],
) -> None:
    """Execute the multi-trial agent test.

    Args:
        config: Test configuration.
        check_fn: The user's check function (receives AgentResult).
    """
    from agentrial.runner.engine import MultiTrialEngine, load_agent
    from agentrial.types import TestCase

    # Load agent
    if config.agent is None:
        raise pytest.skip("No agent configured. Use @agent_test(agent=...) or --agent flag.")

    if isinstance(config.agent, str):
        agent_fn = load_agent(config.agent)
    else:
        agent_fn = config.agent

    # Build a test case
    test_case = TestCase(
        name=check_fn.__name__,
        input=AgentInput(query=config.query, context=config.context),
    )

    # Run trials
    engine = MultiTrialEngine(trials=config.trials, show_progress=False)
    eval_result = engine.run_test_case(agent_fn, test_case)

    # Build AgentResult
    raw_outputs = [t.agent_output for t in eval_result.trials]
    last_output = raw_outputs[-1] if raw_outputs else AgentOutput(output="")

    agent_result = AgentResult(
        output=last_output.output,
        steps=list(last_output.steps),
        cost=eval_result.mean_cost,
        tokens=int(eval_result.mean_tokens),
        duration_ms=eval_result.mean_latency_ms,
        success=all(t.agent_output.success for t in eval_result.trials),
        error=last_output.error,
        pass_rate=eval_result.pass_rate,
        trials_passed=sum(1 for t in eval_result.trials if t.passed),
        trials_total=len(eval_result.trials),
        _raw_outputs=raw_outputs,
    )

    # Run the user's check function
    check_fn(agent_result)

    # Check pass rate threshold
    if agent_result.pass_rate < config.threshold:
        pytest.fail(
            f"Agent pass rate {agent_result.pass_rate:.0%} "
            f"below threshold {config.threshold:.0%} "
            f"({agent_result.trials_passed}/{agent_result.trials_total} trials passed)"
        )


# --- Pytest hooks for --agenteval integration ---


def pytest_addoption(parser: Any) -> None:
    """Add agentrial CLI options to pytest."""
    group = parser.getgroup("agentrial", "Agentrial agent evaluation")
    group.addoption(
        "--agenteval",
        action="store_true",
        default=False,
        help="Enable Agentrial multi-trial agent evaluation",
    )
    group.addoption(
        "--trials",
        type=int,
        default=None,
        help="Override number of trials for @agent_test decorated tests",
    )
    group.addoption(
        "--agent",
        type=str,
        default=None,
        dest="agentrial_agent",
        help="Default agent import path for tests that don't specify one",
    )
    group.addoption(
        "--threshold",
        type=float,
        default=None,
        dest="agentrial_threshold",
        help="Override pass rate threshold for @agent_test decorated tests",
    )


def pytest_configure(config: Any) -> None:
    """Register the agentrial marker."""
    config.addinivalue_line(
        "markers",
        "agentrial: marks a test as an agentrial agent evaluation test",
    )


def pytest_collection_modifyitems(
    config: Any,
    items: list[Any],
) -> None:
    """Apply trial overrides from CLI to agent_test-decorated items."""
    trials_override = config.getoption("--trials", default=None)
    threshold_override = config.getoption("agentrial_threshold", default=None)
    agent_override = config.getoption("agentrial_agent", default=None)

    for item in items:
        # Check if the test function has agentrial config
        fn = getattr(item, "obj", None)
        if fn is None:
            continue

        test_config = getattr(fn, "_agentrial_config", None)
        if test_config is None:
            continue

        # Apply overrides
        if trials_override is not None:
            test_config.trials = trials_override
        if threshold_override is not None:
            test_config.threshold = threshold_override
        if agent_override is not None and test_config.agent is None:
            test_config.agent = agent_override
