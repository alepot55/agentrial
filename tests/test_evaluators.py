"""Tests for evaluators."""

import pytest

from agenteval.evaluators.exact import (
    contains,
    exact_match,
    regex_match,
    tool_called,
)
from agenteval.evaluators.expect import expect
from agenteval.types import (
    AgentMetadata,
    AgentOutput,
    StepType,
    TrajectoryStep,
)


class TestExactMatch:
    """Tests for exact match evaluator."""

    def test_exact_match_true(self) -> None:
        """Test exact match when strings are equal."""
        assert exact_match("hello", "hello")

    def test_exact_match_false(self) -> None:
        """Test exact match when strings differ."""
        assert not exact_match("hello", "world")

    def test_exact_match_case_sensitive(self) -> None:
        """Test case sensitive matching."""
        assert not exact_match("Hello", "hello", case_sensitive=True)

    def test_exact_match_case_insensitive(self) -> None:
        """Test case insensitive matching."""
        assert exact_match("Hello", "hello", case_sensitive=False)


class TestContains:
    """Tests for contains evaluator."""

    def test_contains_all(self) -> None:
        """Test when all substrings are present."""
        passed, missing = contains("The quick brown fox", ["quick", "fox"])
        assert passed
        assert missing == []

    def test_contains_missing(self) -> None:
        """Test when some substrings are missing."""
        passed, missing = contains("The quick brown fox", ["quick", "cat"])
        assert not passed
        assert "cat" in missing

    def test_contains_case_sensitive(self) -> None:
        """Test case sensitive contains."""
        passed, _ = contains("Hello World", ["hello"], case_sensitive=True)
        assert not passed

    def test_contains_case_insensitive(self) -> None:
        """Test case insensitive contains."""
        passed, _ = contains("Hello World", ["hello"], case_sensitive=False)
        assert passed


class TestRegexMatch:
    """Tests for regex match evaluator."""

    def test_regex_match_simple(self) -> None:
        """Test simple regex match."""
        passed, match = regex_match("The price is $19.99", r"\$\d+\.\d+")
        assert passed
        assert match is not None
        assert match.group() == "$19.99"

    def test_regex_no_match(self) -> None:
        """Test regex that doesn't match."""
        passed, match = regex_match("No numbers here", r"\d+")
        assert not passed
        assert match is None

    def test_regex_invalid_pattern(self) -> None:
        """Test invalid regex pattern."""
        passed, match = regex_match("test", r"[invalid")
        assert not passed


class TestToolCalled:
    """Tests for tool_called evaluator."""

    def test_tool_called_found(self) -> None:
        """Test when tool is found."""
        steps = [
            TrajectoryStep(
                step_index=0,
                step_type=StepType.TOOL_CALL,
                name="search",
                parameters={"query": "test"},
            )
        ]
        found, error = tool_called(steps, "search")
        assert found
        assert error is None

    def test_tool_called_not_found(self) -> None:
        """Test when tool is not found."""
        steps = [
            TrajectoryStep(
                step_index=0,
                step_type=StepType.TOOL_CALL,
                name="other_tool",
                parameters={},
            )
        ]
        found, error = tool_called(steps, "search")
        assert not found
        assert "not called" in error

    def test_tool_called_with_params(self) -> None:
        """Test tool called with parameter check."""
        steps = [
            TrajectoryStep(
                step_index=0,
                step_type=StepType.TOOL_CALL,
                name="search",
                parameters={"query": "test", "limit": 10},
            )
        ]
        found, error = tool_called(steps, "search", {"query": "test"})
        assert found

    def test_tool_called_wrong_params(self) -> None:
        """Test tool called with wrong parameters."""
        steps = [
            TrajectoryStep(
                step_index=0,
                step_type=StepType.TOOL_CALL,
                name="search",
                parameters={"query": "wrong"},
            )
        ]
        found, error = tool_called(steps, "search", {"query": "test"})
        assert not found


class TestExpectAPI:
    """Tests for the expect() fluent API."""

    def create_output(
        self,
        output: str = "test output",
        steps: list[TrajectoryStep] | None = None,
        cost: float = 0.01,
        latency_ms: float = 100.0,
        tokens: int = 50,
    ) -> AgentOutput:
        """Create a test AgentOutput."""
        return AgentOutput(
            output=output,
            steps=steps or [],
            metadata=AgentMetadata(
                total_tokens=tokens,
                cost=cost,
                duration_ms=latency_ms,
            ),
        )

    def test_expect_output_contains(self) -> None:
        """Test expect().output.contains()."""
        output = self.create_output("The flight costs $500")
        expectation = expect(output).output.contains("flight", "$500")
        assert expect(output).passed()

    def test_expect_output_missing(self) -> None:
        """Test expect().output.contains() with missing text."""
        output = self.create_output("The flight costs $500")
        e = expect(output)
        e.output.contains("train")
        assert not e.passed()
        assert "train" in str(e.get_failures())

    def test_expect_cost_below(self) -> None:
        """Test expect().cost_below()."""
        output = self.create_output(cost=0.05)
        e = expect(output)
        e.cost_below(0.10)
        assert e.passed()

    def test_expect_cost_exceeded(self) -> None:
        """Test expect().cost_below() when exceeded."""
        output = self.create_output(cost=0.15)
        e = expect(output)
        e.cost_below(0.10)
        assert not e.passed()

    def test_expect_tool_called(self) -> None:
        """Test expect().tool_called()."""
        steps = [
            TrajectoryStep(
                step_index=0,
                step_type=StepType.TOOL_CALL,
                name="search_flights",
                parameters={"origin": "FCO"},
            )
        ]
        output = self.create_output(steps=steps)
        e = expect(output)
        e.tool_called("search_flights")
        assert e.passed()

    def test_expect_step_params(self) -> None:
        """Test expect().step().params_contain()."""
        steps = [
            TrajectoryStep(
                step_index=0,
                step_type=StepType.TOOL_CALL,
                name="search_flights",
                parameters={"origin": "FCO", "destination": "TYO"},
            )
        ]
        output = self.create_output(steps=steps)
        e = expect(output)
        e.step(0).params_contain(origin="FCO")
        assert e.passed()

    def test_expect_chaining(self) -> None:
        """Test chaining multiple expectations."""
        steps = [
            TrajectoryStep(
                step_index=0,
                step_type=StepType.TOOL_CALL,
                name="search",
                parameters={},
            )
        ]
        output = self.create_output(
            output="Found 3 results",
            steps=steps,
            cost=0.05,
        )
        e = expect(output)
        e.output.contains("results")
        e.cost_below(0.10)
        e.tool_called("search")
        assert e.passed()
