"""Multi-trial execution engine."""

import importlib
import logging
import time
from collections.abc import Callable
from typing import Protocol

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from agentrial.evaluators.step_eval import evaluate_output, evaluate_step
from agentrial.metrics.basic import compute_basic_metrics
from agentrial.metrics.statistical import (
    bootstrap_confidence_interval,
    wilson_confidence_interval,
)
from agentrial.metrics.trajectory import attribute_failures
from agentrial.types import (
    AgentInput,
    AgentOutput,
    EvalResult,
    Suite,
    SuiteResult,
    TestCase,
    TrialResult,
)

logger = logging.getLogger(__name__)


class AgentProtocol(Protocol):
    """Protocol for agent callables."""

    def __call__(self, input: AgentInput) -> AgentOutput: ...


def load_agent(agent_path: str) -> Callable[[AgentInput], AgentOutput]:
    """Load an agent function from a Python import path.

    Args:
        agent_path: Dotted import path (e.g., "my_agent.flight_search").

    Returns:
        The agent callable.

    Raises:
        ImportError: If the module or function cannot be found.
    """
    parts = agent_path.rsplit(".", 1)
    if len(parts) == 1:
        raise ImportError(f"Invalid agent path: {agent_path}. Expected 'module.function'")

    module_path, func_name = parts
    module = importlib.import_module(module_path)
    agent = getattr(module, func_name, None)

    if agent is None:
        raise ImportError(f"Function '{func_name}' not found in module '{module_path}'")

    return agent


class MultiTrialEngine:
    """Engine for running multiple trials of agent evaluation.

    This is the core execution engine that runs an agent multiple times
    for each test case and collects results.
    """

    def __init__(
        self,
        trials: int = 10,
        parallel: int = 1,
        show_progress: bool = True,
    ) -> None:
        """Initialize the engine.

        Args:
            trials: Number of trials per test case.
            parallel: Number of parallel workers (not implemented in MVP).
            show_progress: Whether to show progress bars during execution.
        """
        self.trials = trials
        self.parallel = parallel
        self.show_progress = show_progress
        self._progress: Progress | None = None
        self._current_task_id: int | None = None
        self._current_test_index: int = 0
        self._total_tests: int = 0

    def run_single_trial(
        self,
        agent: Callable[[AgentInput], AgentOutput],
        test_case: TestCase,
        trial_index: int,
    ) -> TrialResult:
        """Run a single trial of a test case.

        Args:
            agent: The agent callable.
            test_case: The test case to run.
            trial_index: Index of this trial (0-based).

        Returns:
            TrialResult with pass/fail and metrics.
        """
        start_time = time.time()
        failures: list[str] = []

        try:
            output = agent(test_case.input)
        except Exception as e:
            logger.warning("Agent raised exception in trial %d: %s", trial_index, e)
            from agentrial.types import AgentMetadata

            output = AgentOutput(
                output="",
                steps=[],
                metadata=AgentMetadata(),
                success=False,
                error=str(e),
            )
            failures.append(f"Agent exception: {e}")

        duration_ms = (time.time() - start_time) * 1000

        # Check if agent reported failure
        if not output.success:
            error_msg = output.error or "Agent returned success=False"
            failures.append(f"Agent failed: {error_msg}")

        # Evaluate output if expectations exist and agent succeeded
        if output.success and test_case.expected:
            output_failures = evaluate_output(output, test_case.expected)
            failures.extend(output_failures)

        # Evaluate step expectations
        for step_exp in test_case.step_expectations:
            if step_exp.step_index is not None and step_exp.step_index < len(output.steps):
                step = output.steps[step_exp.step_index]
                step_failures = evaluate_step(step, step_exp)
                failures.extend(step_failures)

        # Check cost constraint
        if test_case.max_cost is not None and output.metadata.cost > test_case.max_cost:
            failures.append(
                f"Cost {output.metadata.cost:.4f} exceeds max {test_case.max_cost:.4f}"
            )

        # Check latency constraint
        if test_case.max_latency_ms is not None and duration_ms > test_case.max_latency_ms:
            failures.append(
                f"Latency {duration_ms:.0f}ms exceeds max {test_case.max_latency_ms:.0f}ms"
            )

        passed = len(failures) == 0

        return TrialResult(
            trial_index=trial_index,
            test_case=test_case,
            agent_output=output,
            passed=passed,
            failures=failures,
            duration_ms=duration_ms,
            cost=output.metadata.cost,
            tokens=output.metadata.total_tokens,
        )

    def run_test_case(
        self,
        agent: Callable[[AgentInput], AgentOutput],
        test_case: TestCase,
    ) -> EvalResult:
        """Run all trials for a test case and compute statistics.

        Args:
            agent: The agent callable.
            test_case: The test case to evaluate.

        Returns:
            EvalResult with pass rate, CI, and metrics.
        """
        trials: list[TrialResult] = []

        for i in range(self.trials):
            logger.debug("Running trial %d/%d for %s", i + 1, self.trials, test_case.name)
            trial = self.run_single_trial(agent, test_case, i)
            trials.append(trial)
            # Update progress bar if active
            if self._progress is not None and self._current_task_id is not None:
                # Format: [test 3/10] test-name trial 5/10
                test_progress = (
                    f"[test {self._current_test_index}/{self._total_tests}]"
                    if self._total_tests > 0 else ""
                )
                self._progress.update(
                    self._current_task_id,
                    advance=1,
                    description=f"{test_progress} {test_case.name} trial {i + 1}/{self.trials}",
                )

        # Compute basic metrics
        metrics = compute_basic_metrics(trials)

        # Compute pass rate and CI
        successes = sum(1 for t in trials if t.passed)
        pass_rate = successes / len(trials)
        pass_rate_ci = wilson_confidence_interval(successes, len(trials))

        # Compute cost CI via bootstrap
        costs = [t.cost for t in trials]
        cost_ci = bootstrap_confidence_interval(costs)

        # Compute latency CI via bootstrap
        latencies = [t.duration_ms for t in trials]
        latency_ci = bootstrap_confidence_interval(latencies)

        # Attribute failures if any
        failure_attribution = None
        if successes < len(trials):
            failure_attribution = attribute_failures(trials)

        return EvalResult(
            test_case=test_case,
            trials=trials,
            pass_rate=pass_rate,
            pass_rate_ci=pass_rate_ci,
            mean_cost=metrics["mean_cost"],
            cost_ci=cost_ci,
            mean_latency_ms=metrics["mean_latency_ms"],
            latency_ci=latency_ci,
            mean_tokens=metrics["mean_tokens"],
            failure_attribution=failure_attribution,
        )

    def run_suite(
        self,
        agent: Callable[[AgentInput], AgentOutput],
        suite: Suite,
    ) -> SuiteResult:
        """Run all test cases in a suite.

        Args:
            agent: The agent callable.
            suite: The test suite to run.

        Returns:
            SuiteResult with all results and overall metrics.
        """
        start_time = time.time()
        results: list[EvalResult] = []

        total_trials = len(suite.cases) * self.trials

        if self.show_progress:
            with Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                transient=True,
            ) as progress:
                self._progress = progress
                self._total_tests = len(suite.cases)
                self._current_task_id = progress.add_task(
                    f"Running {suite.name}",
                    total=total_trials,
                )
                for idx, test_case in enumerate(suite.cases):
                    self._current_test_index = idx + 1
                    logger.info("Evaluating test case: %s", test_case.name)
                    result = self.run_test_case(agent, test_case)
                    results.append(result)
                self._progress = None
                self._current_task_id = None
                self._current_test_index = 0
                self._total_tests = 0
        else:
            for test_case in suite.cases:
                logger.info("Evaluating test case: %s", test_case.name)
                result = self.run_test_case(agent, test_case)
                results.append(result)

        total_duration_ms = (time.time() - start_time) * 1000
        total_cost = sum(r.mean_cost * len(r.trials) for r in results)

        # Compute overall pass rate
        total_trials = sum(len(r.trials) for r in results)
        total_passed = sum(sum(1 for t in r.trials if t.passed) for r in results)
        overall_pass_rate = total_passed / total_trials if total_trials > 0 else 0.0
        overall_ci = wilson_confidence_interval(total_passed, total_trials)

        passed = overall_pass_rate >= suite.threshold

        return SuiteResult(
            suite=suite,
            results=results,
            overall_pass_rate=overall_pass_rate,
            overall_pass_rate_ci=overall_ci,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
            passed=passed,
        )


def run_suite(
    suite: Suite,
    trials: int | None = None,
    parallel: int = 1,
) -> SuiteResult:
    """Convenience function to run a test suite.

    Args:
        suite: The test suite to run.
        trials: Override number of trials (uses suite default if None).
        parallel: Number of parallel workers.

    Returns:
        SuiteResult with all results.
    """
    effective_trials = trials if trials is not None else suite.trials
    engine = MultiTrialEngine(trials=effective_trials, parallel=parallel)

    # Load the agent
    agent = load_agent(suite.agent)

    return engine.run_suite(agent, suite)
