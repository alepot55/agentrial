"""Functional evaluators using custom Python functions."""

from collections.abc import Callable

from agentrial.types import AgentOutput, TrajectoryStep


def custom_output_check(
    output: AgentOutput,
    check_fn: Callable[[AgentOutput], bool],
) -> tuple[bool, str]:
    """Run a custom check function on the agent output.

    Args:
        output: The agent's output.
        check_fn: Custom function that returns True if check passes.

    Returns:
        Tuple of (passed, error_message).
    """
    try:
        result = check_fn(output)
        if result:
            return True, ""
        return False, "Custom check failed"
    except Exception as e:
        return False, f"Custom check raised exception: {e}"


def custom_step_check(
    step: TrajectoryStep,
    check_fn: Callable[[TrajectoryStep], bool],
) -> tuple[bool, str]:
    """Run a custom check function on a trajectory step.

    Args:
        step: The trajectory step.
        check_fn: Custom function that returns True if check passes.

    Returns:
        Tuple of (passed, error_message).
    """
    try:
        result = check_fn(step)
        if result:
            return True, ""
        return False, "Custom step check failed"
    except Exception as e:
        return False, f"Custom step check raised exception: {e}"


def custom_trajectory_check(
    steps: list[TrajectoryStep],
    check_fn: Callable[[list[TrajectoryStep]], bool],
) -> tuple[bool, str]:
    """Run a custom check function on the full trajectory.

    Args:
        steps: List of trajectory steps.
        check_fn: Custom function that returns True if check passes.

    Returns:
        Tuple of (passed, error_message).
    """
    try:
        result = check_fn(steps)
        if result:
            return True, ""
        return False, "Custom trajectory check failed"
    except Exception as e:
        return False, f"Custom trajectory check raised exception: {e}"


def value_in_range(
    value: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> tuple[bool, str]:
    """Check if a value is within a specified range.

    Args:
        value: The value to check.
        min_value: Minimum allowed value (inclusive).
        max_value: Maximum allowed value (inclusive).

    Returns:
        Tuple of (in_range, error_message).
    """
    if min_value is not None and value < min_value:
        return False, f"Value {value} is below minimum {min_value}"
    if max_value is not None and value > max_value:
        return False, f"Value {value} is above maximum {max_value}"
    return True, ""


def output_length_check(
    output: str,
    min_length: int | None = None,
    max_length: int | None = None,
) -> tuple[bool, str]:
    """Check if output length is within bounds.

    Args:
        output: The output string.
        min_length: Minimum allowed length.
        max_length: Maximum allowed length.

    Returns:
        Tuple of (in_bounds, error_message).
    """
    length = len(output)

    if min_length is not None and length < min_length:
        return False, f"Output length {length} is below minimum {min_length}"
    if max_length is not None and length > max_length:
        return False, f"Output length {length} is above maximum {max_length}"
    return True, ""


def semantic_similarity(
    actual: str,
    expected: str,
    threshold: float = 0.8,
    model: str | None = None,
) -> tuple[bool, float]:
    """Check semantic similarity between actual and expected output.

    Note: This is a placeholder for future LLM-as-judge integration.
    For MVP, this falls back to simple token overlap.

    Args:
        actual: The actual output.
        expected: The expected output.
        threshold: Minimum similarity score (0-1).
        model: Optional model to use for comparison.

    Returns:
        Tuple of (above_threshold, similarity_score).
    """
    # Simple token overlap for MVP
    actual_tokens = set(actual.lower().split())
    expected_tokens = set(expected.lower().split())

    if not expected_tokens:
        return True, 1.0

    intersection = actual_tokens & expected_tokens
    similarity = len(intersection) / len(expected_tokens)

    return similarity >= threshold, similarity
