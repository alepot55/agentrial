"""Exact match, contains, and regex evaluators."""

import re
from typing import Any


def exact_match(actual: str, expected: str, case_sensitive: bool = True) -> bool:
    """Check if actual output exactly matches expected.

    Args:
        actual: The actual output string.
        expected: The expected output string.
        case_sensitive: Whether comparison is case-sensitive.

    Returns:
        True if strings match.
    """
    if not case_sensitive:
        return actual.lower() == expected.lower()
    return actual == expected


def contains(
    actual: str, substrings: list[str], case_sensitive: bool = True
) -> tuple[bool, list[str]]:
    """Check if actual output contains all expected substrings (AND logic).

    Args:
        actual: The actual output string.
        substrings: List of substrings that must all be present.
        case_sensitive: Whether comparison is case-sensitive.

    Returns:
        Tuple of (all_present, missing_substrings).
    """
    check_actual = actual if case_sensitive else actual.lower()
    missing = []

    for substring in substrings:
        check_substring = substring if case_sensitive else substring.lower()
        if check_substring not in check_actual:
            missing.append(substring)

    return len(missing) == 0, missing


def contains_any(
    actual: str, substrings: list[str], case_sensitive: bool = True
) -> tuple[bool, str | None]:
    """Check if actual output contains at least one of the expected substrings (OR logic).

    Args:
        actual: The actual output string.
        substrings: List of substrings, at least one must be present.
        case_sensitive: Whether comparison is case-sensitive.

    Returns:
        Tuple of (found_any, matched_substring or None).
    """
    check_actual = actual if case_sensitive else actual.lower()

    for substring in substrings:
        check_substring = substring if case_sensitive else substring.lower()
        if check_substring in check_actual:
            return True, substring

    return False, None


def regex_match(actual: str, pattern: str, flags: int = 0) -> tuple[bool, re.Match[str] | None]:
    """Check if actual output matches a regex pattern.

    Args:
        actual: The actual output string.
        pattern: Regex pattern to match.
        flags: Regex flags (e.g., re.IGNORECASE).

    Returns:
        Tuple of (matches, match_object).
    """
    try:
        match = re.search(pattern, actual, flags)
        return match is not None, match
    except re.error:
        return False, None


def tool_called(
    steps: list[Any],
    tool_name: str,
    params_contain: dict[str, Any] | None = None,
) -> tuple[bool, str | None]:
    """Check if a specific tool was called in the trajectory.

    Args:
        steps: List of trajectory steps.
        tool_name: Name of the tool to look for.
        params_contain: Optional parameters that must be present.

    Returns:
        Tuple of (found, error_message).
    """
    from agentrial.types import StepType

    for step in steps:
        if step.step_type == StepType.TOOL_CALL and step.name == tool_name:
            # Check parameters if specified
            if params_contain:
                missing_params = []
                for key, expected_value in params_contain.items():
                    actual_value = step.parameters.get(key)
                    if actual_value != expected_value:
                        missing_params.append(f"{key}={expected_value}")

                if missing_params:
                    return False, f"Tool {tool_name} called but missing params: {missing_params}"

            return True, None

    return False, f"Tool {tool_name} was not called"


def output_format_json(actual: str) -> tuple[bool, Any]:
    """Check if output is valid JSON and return parsed value.

    Args:
        actual: The actual output string.

    Returns:
        Tuple of (is_valid_json, parsed_or_error).
    """
    import json

    try:
        parsed = json.loads(actual)
        return True, parsed
    except json.JSONDecodeError as e:
        return False, str(e)


def output_format_number(actual: str) -> tuple[bool, float | None]:
    """Check if output contains a number and extract it.

    Args:
        actual: The actual output string.

    Returns:
        Tuple of (found_number, extracted_number).
    """
    # Try to find a number in the output
    match = re.search(r"-?\d+\.?\d*", actual)
    if match:
        try:
            return True, float(match.group())
        except ValueError:
            pass
    return False, None
