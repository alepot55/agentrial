"""Tests for snapshot testing."""

import tempfile
from pathlib import Path

from agentrial.snapshots import (
    CaseComparison,
    compare_with_snapshot,
    create_snapshot,
    load_snapshot,
    save_snapshot,
)
from agentrial.types import (
    AgentInput,
    AgentMetadata,
    AgentOutput,
    ConfidenceInterval,
    EvalResult,
    StepType,
    Suite,
    SuiteResult,
    TestCase,
    TrajectoryStep,
    TrialResult,
)


def _make_suite_result(
    pass_rates: list[float] | None = None,
    trials_per_case: int = 10,
) -> SuiteResult:
    """Create a suite result for testing."""
    if pass_rates is None:
        pass_rates = [0.8, 1.0]

    suite = Suite(name="test-suite", agent="test.agent", trials=trials_per_case)
    results = []

    for i, rate in enumerate(pass_rates):
        pass_count = int(rate * trials_per_case)
        fail_count = trials_per_case - pass_count

        trials = []
        for j in range(pass_count):
            trials.append(
                TrialResult(
                    trial_index=j,
                    test_case=TestCase(name=f"case-{i}", input=AgentInput(query="q")),
                    agent_output=AgentOutput(
                        output="ok",
                        steps=[
                            TrajectoryStep(0, StepType.TOOL_CALL, "search", duration_ms=50),
                        ],
                        metadata=AgentMetadata(cost=0.01, duration_ms=100),
                    ),
                    passed=True,
                    duration_ms=100.0,
                    cost=0.01,
                )
            )
        for j in range(fail_count):
            trials.append(
                TrialResult(
                    trial_index=pass_count + j,
                    test_case=TestCase(name=f"case-{i}", input=AgentInput(query="q")),
                    agent_output=AgentOutput(
                        output="fail",
                        steps=[
                            TrajectoryStep(0, StepType.TOOL_CALL, "wrong", duration_ms=50),
                        ],
                        metadata=AgentMetadata(cost=0.02, duration_ms=200),
                    ),
                    passed=False,
                    failures=["assertion failed"],
                    duration_ms=200.0,
                    cost=0.02,
                )
            )

        results.append(
            EvalResult(
                test_case=TestCase(name=f"case-{i}", input=AgentInput(query="q")),
                trials=trials,
                pass_rate=rate,
                pass_rate_ci=ConfidenceInterval(lower=rate - 0.15, upper=min(rate + 0.15, 1.0)),
                mean_cost=0.01 * rate + 0.02 * (1 - rate),
                cost_ci=ConfidenceInterval(lower=0.008, upper=0.025),
                mean_latency_ms=100 * rate + 200 * (1 - rate),
                latency_ci=ConfidenceInterval(lower=80, upper=220),
                mean_tokens=0,
            )
        )

    total_passed = sum(
        sum(1 for t in r.trials if t.passed) for r in results
    )
    total_trials = sum(len(r.trials) for r in results)
    overall_rate = total_passed / total_trials if total_trials else 0

    return SuiteResult(
        suite=suite,
        results=results,
        overall_pass_rate=overall_rate,
        overall_pass_rate_ci=ConfidenceInterval(lower=overall_rate - 0.1, upper=overall_rate + 0.1),
        total_cost=sum(r.mean_cost * len(r.trials) for r in results),
        total_duration_ms=1000,
        passed=overall_rate >= 0.85,
    )


class TestCreateSnapshot:
    """Tests for create_snapshot."""

    def test_basic_structure(self) -> None:
        result = _make_suite_result()
        snap = create_snapshot(result)

        assert snap["version"] == "1.0"
        assert snap["suite"] == "test-suite"
        assert "created_at" in snap
        assert len(snap["cases"]) == 2

    def test_case_metrics(self) -> None:
        result = _make_suite_result(pass_rates=[0.8])
        snap = create_snapshot(result)

        case = snap["cases"][0]
        assert case["name"] == "case-0"
        assert case["metrics"]["pass_rate"]["mean"] == 0.8
        assert len(case["metrics"]["cost"]["values"]) == 10
        assert len(case["metrics"]["latency"]["values"]) == 10

    def test_step_distributions(self) -> None:
        result = _make_suite_result(pass_rates=[0.8])
        snap = create_snapshot(result)

        case = snap["cases"][0]
        steps = case["step_distributions"]
        assert "step_0" in steps
        # 8 trials have "search", 2 have "wrong"
        assert steps["step_0"]["top_actions"]["search"] == 0.8
        assert steps["step_0"]["top_actions"]["wrong"] == 0.2

    def test_overall_metrics(self) -> None:
        result = _make_suite_result(pass_rates=[1.0, 1.0])
        snap = create_snapshot(result)

        assert snap["overall"]["pass_rate"] == 1.0


class TestSaveLoadSnapshot:
    """Tests for saving and loading snapshots."""

    def test_save_and_load(self) -> None:
        result = _make_suite_result()
        snap = create_snapshot(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            save_snapshot(snap, path=path)

            loaded = load_snapshot(path)
            assert loaded["suite"] == snap["suite"]
            assert loaded["cases"][0]["name"] == snap["cases"][0]["name"]

    def test_save_default_path(self) -> None:
        result = _make_suite_result()
        snap = create_snapshot(result)

        with tempfile.TemporaryDirectory() as tmpdir:
            import os

            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                path = save_snapshot(snap)
                assert path.exists()
                assert "test-suite" in str(path)
            finally:
                os.chdir(old_cwd)

    def test_load_nonexistent(self) -> None:
        import pytest

        with pytest.raises(FileNotFoundError):
            load_snapshot("/nonexistent/path.json")


class TestCompareWithSnapshot:
    """Tests for snapshot comparison."""

    def test_no_change(self) -> None:
        result = _make_suite_result(pass_rates=[0.8, 1.0])
        snap = create_snapshot(result)

        comparison = compare_with_snapshot(result, snap)
        assert comparison.overall_passed
        assert len(comparison.regressions) == 0

    def test_regression_detected(self) -> None:
        # Create baseline with high pass rate
        baseline = _make_suite_result(pass_rates=[1.0], trials_per_case=20)
        snap = create_snapshot(baseline)

        # Create current with lower pass rate
        current = _make_suite_result(pass_rates=[0.3], trials_per_case=20)

        comparison = compare_with_snapshot(current, snap)
        assert not comparison.overall_passed
        assert len(comparison.regressions) >= 1
        assert comparison.cases[0].status == "regression"

    def test_improvement_detected(self) -> None:
        # Baseline with low pass rate
        baseline = _make_suite_result(pass_rates=[0.3], trials_per_case=20)
        snap = create_snapshot(baseline)

        # Current with high pass rate
        current = _make_suite_result(pass_rates=[1.0], trials_per_case=20)

        comparison = compare_with_snapshot(current, snap)
        assert comparison.overall_passed
        assert len(comparison.improvements) >= 1

    def test_new_case(self) -> None:
        baseline = _make_suite_result(pass_rates=[0.8])
        snap = create_snapshot(baseline)

        current = _make_suite_result(pass_rates=[0.8, 1.0])

        comparison = compare_with_snapshot(current, snap)
        new_cases = [c for c in comparison.cases if c.status == "new"]
        assert len(new_cases) == 1
        assert new_cases[0].name == "case-1"

    def test_removed_case(self) -> None:
        baseline = _make_suite_result(pass_rates=[0.8, 1.0])
        snap = create_snapshot(baseline)

        current = _make_suite_result(pass_rates=[0.8])

        comparison = compare_with_snapshot(current, snap)
        removed = [c for c in comparison.cases if c.status == "removed"]
        assert len(removed) == 1
        assert removed[0].name == "case-1"


class TestCaseComparison:
    """Tests for CaseComparison."""

    def test_is_regression(self) -> None:
        c = CaseComparison(name="test", status="regression")
        assert c.is_regression

        c = CaseComparison(name="test", status="cost_regression")
        assert c.is_regression

        c = CaseComparison(name="test", status="no_change")
        assert not c.is_regression
