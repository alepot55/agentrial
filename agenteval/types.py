"""Core type definitions for AgentEval."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class StepType(Enum):
    """Type of trajectory step."""

    TOOL_CALL = "tool_call"
    LLM_CALL = "llm_call"
    OBSERVATION = "observation"
    REASONING = "reasoning"
    OUTPUT = "output"


@dataclass
class TrajectoryStep:
    """A single step in an agent's trajectory.

    Attributes:
        step_index: Zero-based index of this step in the trajectory.
        step_type: The type of step (tool_call, llm_call, etc.).
        name: Name of the action (e.g., tool name, "llm_response").
        parameters: Input parameters for this step.
        output: Output/result from this step.
        duration_ms: Time taken for this step in milliseconds.
        tokens: Number of tokens used in this step.
        metadata: Additional step-specific metadata.
    """

    step_index: int
    step_type: StepType
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    duration_ms: float = 0.0
    tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentInput:
    """Input to an agent.

    Attributes:
        query: The main input query/prompt.
        context: Additional context for the agent.
    """

    query: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMetadata:
    """Metadata about an agent execution.

    Attributes:
        total_tokens: Total tokens used across all steps.
        prompt_tokens: Tokens used for prompts.
        completion_tokens: Tokens used for completions.
        cost: Estimated cost in USD.
        duration_ms: Total execution time in milliseconds.
    """

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    duration_ms: float = 0.0


@dataclass
class AgentOutput:
    """Output from an agent execution.

    Attributes:
        output: The final output string from the agent.
        steps: Ordered list of trajectory steps.
        metadata: Execution metadata (tokens, cost, duration).
        success: Whether the agent completed without errors.
        error: Error message if the agent failed.
    """

    output: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    metadata: AgentMetadata = field(default_factory=AgentMetadata)
    success: bool = True
    error: str | None = None


@dataclass
class StepExpectation:
    """Expected behavior for a trajectory step.

    Attributes:
        step_index: Which step this applies to (0-based).
        name: Optional expected step name (e.g., expected tool name).
        expected_tool: Expected tool to be called.
        params_contain: Parameters that must be present.
        output_contains: Strings that must appear in output.
        custom_check: Custom validation function.
    """

    step_index: int | None = None
    name: str | None = None
    expected_tool: str | None = None
    params_contain: dict[str, Any] | None = None
    output_contains: list[str] | None = None
    custom_check: Callable[[TrajectoryStep], bool] | None = None


@dataclass
class ExpectedOutput:
    """Expected output criteria for a test case.

    Attributes:
        exact_match: Output must match this exactly.
        contains: Output must contain all these strings.
        regex: Output must match this regex pattern.
        tool_calls: Expected tool calls in order.
        custom_check: Custom validation function.
    """

    exact_match: str | None = None
    contains: list[str] | None = None
    regex: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    custom_check: Callable[[AgentOutput], bool] | None = None


@dataclass
class TestCase:
    """A single test case for agent evaluation.

    Attributes:
        name: Unique name for this test case.
        input: The input to provide to the agent.
        expected: Expected output criteria.
        step_expectations: Per-step expectations.
        max_cost: Maximum allowed cost per run.
        max_latency_ms: Maximum allowed latency in milliseconds.
        tags: Optional tags for filtering.
    """

    name: str
    input: AgentInput
    expected: ExpectedOutput | None = None
    step_expectations: list[StepExpectation] = field(default_factory=list)
    max_cost: float | None = None
    max_latency_ms: float | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class Suite:
    """A test suite containing multiple test cases.

    Attributes:
        name: Suite name.
        agent: Python import path to the agent function.
        trials: Number of trials per test case.
        threshold: Minimum pass rate to consider the suite passing.
        cases: List of test cases.
        tags: Optional tags for filtering.
    """

    name: str
    agent: str
    trials: int = 10
    threshold: float = 0.85
    cases: list[TestCase] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def case(self, func: Callable[..., None]) -> Callable[..., None]:
        """Decorator to add a test case to the suite.

        The decorated function receives the AgentOutput and should use
        expect() to make assertions.
        """
        # Create a TestCase from the function
        test_case = TestCase(
            name=func.__name__,
            input=AgentInput(query=""),  # Will be set by the function
        )
        test_case.metadata = {"validator": func}  # type: ignore
        self.cases.append(test_case)
        return func


@dataclass
class TrialResult:
    """Result of a single trial (one agent execution).

    Attributes:
        trial_index: Zero-based trial index.
        test_case: The test case that was run.
        agent_output: The agent's output.
        passed: Whether all expectations were met.
        failures: List of failure messages.
        duration_ms: Time taken for this trial.
        cost: Cost of this trial.
        tokens: Tokens used in this trial.
    """

    trial_index: int
    test_case: TestCase
    agent_output: AgentOutput
    passed: bool
    failures: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    cost: float = 0.0
    tokens: int = 0


@dataclass
class ConfidenceInterval:
    """A confidence interval.

    Attributes:
        lower: Lower bound.
        upper: Upper bound.
        confidence_level: Confidence level (e.g., 0.95 for 95%).
    """

    lower: float
    upper: float
    confidence_level: float = 0.95


@dataclass
class EvalResult:
    """Result of evaluating a single test case across multiple trials.

    Attributes:
        test_case: The test case that was evaluated.
        trials: Results from each trial.
        pass_rate: Fraction of trials that passed.
        pass_rate_ci: Confidence interval for pass rate.
        mean_cost: Mean cost across trials.
        cost_ci: Confidence interval for cost.
        mean_latency_ms: Mean latency across trials.
        latency_ci: Confidence interval for latency.
        mean_tokens: Mean tokens across trials.
        failure_attribution: Which step most likely caused failures.
    """

    test_case: TestCase
    trials: list[TrialResult]
    pass_rate: float
    pass_rate_ci: ConfidenceInterval
    mean_cost: float
    cost_ci: ConfidenceInterval
    mean_latency_ms: float
    latency_ci: ConfidenceInterval
    mean_tokens: float
    failure_attribution: dict[str, Any] | None = None


@dataclass
class SuiteResult:
    """Result of evaluating an entire test suite.

    Attributes:
        suite: The suite that was evaluated.
        results: Results for each test case.
        overall_pass_rate: Overall pass rate across all cases.
        overall_pass_rate_ci: Confidence interval for overall pass rate.
        total_cost: Total cost of all trials.
        total_duration_ms: Total time taken.
        passed: Whether the suite passed (overall_pass_rate >= threshold).
    """

    suite: Suite
    results: list[EvalResult]
    overall_pass_rate: float
    overall_pass_rate_ci: ConfidenceInterval
    total_cost: float
    total_duration_ms: float
    passed: bool
