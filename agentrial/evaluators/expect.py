"""Fluent assertion API for test cases."""

from typing import Any

from agentrial.types import AgentOutput, TrajectoryStep


class ExpectationBuilder:
    """Fluent builder for test expectations.

    Usage:
        expect(result).output.contains("flight")
        expect(result).tool_called("search_flights")
        expect(result).step(2).params_contain(origin="FCO")
        expect(result).cost_below(0.15)
    """

    def __init__(self, output: AgentOutput) -> None:
        """Initialize with agent output.

        Args:
            output: The AgentOutput to make assertions against.
        """
        self._output = output
        self._failures: list[str] = []

    @property
    def output(self) -> "OutputExpectation":
        """Start an output expectation chain."""
        return OutputExpectation(self._output.output, self._failures)

    def tool_called(
        self,
        tool_name: str,
        params_contain: dict[str, Any] | None = None,
    ) -> "ExpectationBuilder":
        """Assert that a specific tool was called.

        Args:
            tool_name: Name of the tool.
            params_contain: Optional parameters that must be present.

        Returns:
            Self for chaining.
        """
        from agentrial.evaluators.exact import tool_called

        passed, error = tool_called(self._output.steps, tool_name, params_contain)
        if not passed and error:
            self._failures.append(error)
        return self

    def step(self, index: int) -> "StepExpectation":
        """Get expectation builder for a specific step.

        Args:
            index: Zero-based step index.

        Returns:
            StepExpectation for the specified step.
        """
        if index >= len(self._output.steps):
            self._failures.append(
                f"Expected step at index {index}, but only {len(self._output.steps)} steps exist"
            )
            # Return a dummy step expectation that records failures
            return StepExpectation(None, self._failures)

        return StepExpectation(self._output.steps[index], self._failures)

    def cost_below(self, max_cost: float) -> "ExpectationBuilder":
        """Assert that cost is below a threshold.

        Args:
            max_cost: Maximum allowed cost in USD.

        Returns:
            Self for chaining.
        """
        if self._output.metadata.cost > max_cost:
            self._failures.append(
                f"Cost {self._output.metadata.cost:.4f} exceeds maximum {max_cost:.4f}"
            )
        return self

    def latency_below(self, max_ms: float) -> "ExpectationBuilder":
        """Assert that latency is below a threshold.

        Args:
            max_ms: Maximum allowed latency in milliseconds.

        Returns:
            Self for chaining.
        """
        if self._output.metadata.duration_ms > max_ms:
            self._failures.append(
                f"Latency {self._output.metadata.duration_ms:.0f}ms exceeds maximum {max_ms:.0f}ms"
            )
        return self

    def tokens_below(self, max_tokens: int) -> "ExpectationBuilder":
        """Assert that token usage is below a threshold.

        Args:
            max_tokens: Maximum allowed tokens.

        Returns:
            Self for chaining.
        """
        if self._output.metadata.total_tokens > max_tokens:
            self._failures.append(
                f"Tokens {self._output.metadata.total_tokens} exceeds maximum {max_tokens}"
            )
        return self

    def succeeded(self) -> "ExpectationBuilder":
        """Assert that the agent execution succeeded.

        Returns:
            Self for chaining.
        """
        if not self._output.success:
            error = self._output.error or "Unknown error"
            self._failures.append(f"Agent execution failed: {error}")
        return self

    def trajectory_length(
        self,
        min_steps: int | None = None,
        max_steps: int | None = None,
    ) -> "ExpectationBuilder":
        """Assert trajectory length is within bounds.

        Args:
            min_steps: Minimum number of steps.
            max_steps: Maximum number of steps.

        Returns:
            Self for chaining.
        """
        actual = len(self._output.steps)
        if min_steps is not None and actual < min_steps:
            self._failures.append(f"Trajectory has {actual} steps, expected at least {min_steps}")
        if max_steps is not None and actual > max_steps:
            self._failures.append(f"Trajectory has {actual} steps, expected at most {max_steps}")
        return self

    def get_failures(self) -> list[str]:
        """Get all recorded failures.

        Returns:
            List of failure messages.
        """
        return self._failures.copy()

    def passed(self) -> bool:
        """Check if all expectations passed.

        Returns:
            True if no failures were recorded.
        """
        return len(self._failures) == 0


class OutputExpectation:
    """Expectation builder for agent output content."""

    def __init__(self, output: str, failures: list[str]) -> None:
        """Initialize with output string.

        Args:
            output: The output string to check.
            failures: Shared failures list.
        """
        self._output = output
        self._failures = failures

    def contains(self, *substrings: str) -> "OutputExpectation":
        """Assert output contains all substrings.

        Args:
            substrings: Substrings that must all be present.

        Returns:
            Self for chaining.
        """
        from agentrial.evaluators.exact import contains

        passed, missing = contains(self._output, list(substrings))
        if not passed:
            self._failures.append(f"Output missing expected substrings: {missing}")
        return self

    def equals(self, expected: str) -> "OutputExpectation":
        """Assert output exactly equals expected.

        Args:
            expected: Expected output string.

        Returns:
            Self for chaining.
        """
        if self._output != expected:
            self._failures.append("Output does not match expected value")
        return self

    def matches(self, pattern: str) -> "OutputExpectation":
        """Assert output matches regex pattern.

        Args:
            pattern: Regex pattern.

        Returns:
            Self for chaining.
        """
        from agentrial.evaluators.exact import regex_match

        passed, _ = regex_match(self._output, pattern)
        if not passed:
            self._failures.append(f"Output does not match pattern '{pattern}'")
        return self

    def length_between(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
    ) -> "OutputExpectation":
        """Assert output length is within bounds.

        Args:
            min_length: Minimum length.
            max_length: Maximum length.

        Returns:
            Self for chaining.
        """
        actual = len(self._output)
        if min_length is not None and actual < min_length:
            self._failures.append(f"Output length {actual} is below minimum {min_length}")
        if max_length is not None and actual > max_length:
            self._failures.append(f"Output length {actual} is above maximum {max_length}")
        return self


class StepExpectation:
    """Expectation builder for a trajectory step."""

    def __init__(self, step: TrajectoryStep | None, failures: list[str]) -> None:
        """Initialize with a step.

        Args:
            step: The trajectory step (may be None if index out of bounds).
            failures: Shared failures list.
        """
        self._step = step
        self._failures = failures

    def tool_name(self, expected: str) -> "StepExpectation":
        """Assert step is a tool call with given name.

        Args:
            expected: Expected tool name.

        Returns:
            Self for chaining.
        """
        if self._step is None:
            return self

        from agentrial.types import StepType

        if self._step.step_type != StepType.TOOL_CALL:
            self._failures.append(
                f"Step {self._step.step_index}: expected tool call, "
                f"got {self._step.step_type.value}"
            )
        elif self._step.name != expected:
            self._failures.append(
                f"Step {self._step.step_index}: expected tool '{expected}', got '{self._step.name}'"
            )
        return self

    def params_contain(self, **expected_params: Any) -> "StepExpectation":
        """Assert step parameters contain expected values.

        Args:
            **expected_params: Key-value pairs that must be present.

        Returns:
            Self for chaining.
        """
        if self._step is None:
            return self

        for key, expected_value in expected_params.items():
            actual_value = self._step.parameters.get(key)
            if actual_value != expected_value:
                self._failures.append(
                    f"Step {self._step.step_index}: param '{key}' expected '{expected_value}', "
                    f"got '{actual_value}'"
                )
        return self

    def output_contains(self, *substrings: str) -> "StepExpectation":
        """Assert step output contains all substrings.

        Args:
            substrings: Substrings that must be present.

        Returns:
            Self for chaining.
        """
        if self._step is None:
            return self

        from agentrial.evaluators.exact import contains

        output_str = str(self._step.output) if self._step.output is not None else ""
        passed, missing = contains(output_str, list(substrings))
        if not passed:
            self._failures.append(
                f"Step {self._step.step_index}: output missing substrings: {missing}"
            )
        return self


def expect(output: AgentOutput) -> ExpectationBuilder:
    """Create an expectation builder for fluent assertions.

    This is the main entry point for the expect API.

    Args:
        output: The AgentOutput to make assertions against.

    Returns:
        ExpectationBuilder for fluent assertions.

    Example:
        expect(result).output.contains("flight")
        expect(result).tool_called("search_flights")
        expect(result).step(0).params_contain(origin="FCO")
        expect(result).cost_below(0.15)
    """
    return ExpectationBuilder(output)
