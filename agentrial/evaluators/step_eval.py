"""Step-level evaluation logic."""

from agentrial.evaluators.exact import contains, contains_any, exact_match, regex_match, tool_called
from agentrial.types import AgentOutput, ExpectedOutput, StepExpectation, StepType, TrajectoryStep


def evaluate_output(output: AgentOutput, expected: ExpectedOutput) -> list[str]:
    """Evaluate agent output against expectations.

    Args:
        output: The agent's output.
        expected: Expected output criteria.

    Returns:
        List of failure messages (empty if all checks pass).
    """
    failures: list[str] = []

    # Check exact match
    if expected.exact_match is not None:
        if not exact_match(output.output, expected.exact_match):
            failures.append(
                f"Exact match failed: expected '{expected.exact_match[:50]}...', "
                f"got '{output.output[:50]}...'"
            )

    # Check contains (AND logic)
    if expected.contains:
        passed, missing = contains(output.output, expected.contains)
        if not passed:
            failures.append(f"Missing expected substrings: {missing}")

    # Check contains_any (OR logic)
    if expected.contains_any:
        passed, _ = contains_any(output.output, expected.contains_any)
        if not passed:
            failures.append(f"Output must contain at least one of: {expected.contains_any}")

    # Check regex
    if expected.regex:
        passed, _ = regex_match(output.output, expected.regex)
        if not passed:
            failures.append(f"Regex '{expected.regex}' did not match output")

    # Check tool calls
    if expected.tool_calls:
        for expected_call in expected.tool_calls:
            tool_name = expected_call.get("tool", expected_call.get("name", ""))
            params = expected_call.get("params_contain", expected_call.get("params", {}))

            passed, error = tool_called(output.steps, tool_name, params)
            if not passed and error:
                failures.append(error)

    # Check custom function
    if expected.custom_check:
        try:
            if not expected.custom_check(output):
                failures.append("Custom check function returned False")
        except Exception as e:
            failures.append(f"Custom check function raised exception: {e}")

    return failures


def evaluate_step(step: TrajectoryStep, expectation: StepExpectation) -> list[str]:
    """Evaluate a single trajectory step against expectations.

    Args:
        step: The trajectory step to evaluate.
        expectation: Expected step behavior.

    Returns:
        List of failure messages (empty if all checks pass).
    """
    failures: list[str] = []

    # Check step name
    if expectation.name and step.name != expectation.name:
        failures.append(
            f"Step {step.step_index}: expected name '{expectation.name}', got '{step.name}'"
        )

    # Check expected tool
    if expectation.expected_tool:
        if step.step_type != StepType.TOOL_CALL:
            failures.append(
                f"Step {step.step_index}: expected tool call, got {step.step_type.value}"
            )
        elif step.name != expectation.expected_tool:
            failures.append(
                f"Step {step.step_index}: expected tool '{expectation.expected_tool}', "
                f"got '{step.name}'"
            )

    # Check parameters contain
    if expectation.params_contain:
        for key, expected_value in expectation.params_contain.items():
            actual_value = step.parameters.get(key)
            if actual_value != expected_value:
                failures.append(
                    f"Step {step.step_index}: parameter '{key}' expected '{expected_value}', "
                    f"got '{actual_value}'"
                )

    # Check output contains
    if expectation.output_contains:
        step_output = str(step.output) if step.output is not None else ""
        passed, missing = contains(step_output, expectation.output_contains)
        if not passed:
            failures.append(f"Step {step.step_index}: output missing: {missing}")

    # Check custom function
    if expectation.custom_check:
        try:
            if not expectation.custom_check(step):
                failures.append(f"Step {step.step_index}: custom check failed")
        except Exception as e:
            failures.append(f"Step {step.step_index}: custom check raised exception: {e}")

    return failures


def evaluate_trajectory(
    steps: list[TrajectoryStep],
    expectations: list[StepExpectation],
) -> list[str]:
    """Evaluate a full trajectory against step expectations.

    Args:
        steps: The trajectory steps.
        expectations: List of step expectations.

    Returns:
        List of failure messages.
    """
    failures: list[str] = []

    for expectation in expectations:
        if expectation.step_index is not None:
            if expectation.step_index >= len(steps):
                failures.append(
                    f"Expected step at index {expectation.step_index}, "
                    f"but trajectory only has {len(steps)} steps"
                )
                continue

            step = steps[expectation.step_index]
            failures.extend(evaluate_step(step, expectation))

    return failures
