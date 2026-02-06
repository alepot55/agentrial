"""Tests for Agent Reliability Score."""

from agentrial.ars import ARSBreakdown, compute_ars, compute_ars_from_json
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
    pass_rate: float = 0.9,
    num_cases: int = 2,
    num_trials: int = 10,
    cost: float = 0.01,
    latency_ms: float = 500.0,
) -> SuiteResult:
    """Create a SuiteResult for testing."""
    suite = Suite(
        name="test-suite",
        agent="test.agent",
        trials=num_trials,
        threshold=0.85,
        cases=[],
        tags=[],
    )

    results = []
    for i in range(num_cases):
        tc = TestCase(
            name=f"test-case-{i}",
            input=AgentInput(query=f"query {i}"),
        )
        trials = []
        for j in range(num_trials):
            passed = j < int(pass_rate * num_trials)
            trials.append(
                TrialResult(
                    trial_index=j,
                    test_case=tc,
                    agent_output=AgentOutput(
                        output=f"output {j}",
                        steps=[
                            TrajectoryStep(
                                step_index=0,
                                step_type=StepType.TOOL_CALL,
                                name="search",
                                parameters={},
                                output="result",
                                duration_ms=latency_ms / 2,
                            ),
                            TrajectoryStep(
                                step_index=1,
                                step_type=StepType.OUTPUT,
                                name="respond",
                                parameters={},
                                output=f"output {j}",
                                duration_ms=latency_ms / 2,
                            ),
                        ],
                        metadata=AgentMetadata(
                            total_tokens=100,
                            cost=cost,
                            duration_ms=latency_ms,
                        ),
                        success=passed,
                    ),
                    passed=passed,
                    duration_ms=latency_ms,
                    cost=cost,
                    tokens=100,
                )
            )
        results.append(
            EvalResult(
                test_case=tc,
                trials=trials,
                pass_rate=pass_rate,
                pass_rate_ci=ConfidenceInterval(lower=0.7, upper=0.98),
                mean_cost=cost,
                cost_ci=ConfidenceInterval(lower=cost * 0.8, upper=cost * 1.2),
                mean_latency_ms=latency_ms,
                latency_ci=ConfidenceInterval(lower=latency_ms * 0.8, upper=latency_ms * 1.2),
                mean_tokens=100.0,
            )
        )

    return SuiteResult(
        suite=suite,
        results=results,
        overall_pass_rate=pass_rate,
        overall_pass_rate_ci=ConfidenceInterval(lower=0.7, upper=0.98),
        total_cost=cost * num_cases * num_trials,
        total_duration_ms=latency_ms * num_cases * num_trials,
        passed=pass_rate >= 0.85,
    )


class TestComputeARS:
    """Test ARS computation from SuiteResult."""

    def test_basic_score(self):
        sr = _make_suite_result(pass_rate=0.9)
        ars = compute_ars(sr)
        assert isinstance(ars, ARSBreakdown)
        assert 0 <= ars.score <= 100

    def test_high_pass_rate_gives_high_score(self):
        sr = _make_suite_result(pass_rate=1.0)
        ars = compute_ars(sr)
        assert ars.score >= 70

    def test_low_pass_rate_gives_low_score(self):
        sr = _make_suite_result(pass_rate=0.1)
        ars = compute_ars(sr)
        # Accuracy is only 40% of the score; other components stay high
        assert ars.accuracy == 10.0
        assert ars.score < 80

    def test_accuracy_component(self):
        sr = _make_suite_result(pass_rate=0.9)
        ars = compute_ars(sr)
        assert ars.accuracy == 90.0

    def test_consistency_single_case(self):
        sr = _make_suite_result(pass_rate=0.9, num_cases=1)
        ars = compute_ars(sr)
        assert ars.consistency == 100.0

    def test_cost_efficiency_low_cost(self):
        sr = _make_suite_result(cost=0.001)
        ars = compute_ars(sr, cost_ceiling=1.0)
        assert ars.cost_efficiency >= 90.0

    def test_cost_efficiency_high_cost(self):
        sr = _make_suite_result(cost=0.9)
        ars = compute_ars(sr, cost_ceiling=1.0)
        assert ars.cost_efficiency < 50.0

    def test_trajectory_quality_with_steps(self):
        sr = _make_suite_result()
        ars = compute_ars(sr)
        # All steps have non-None output
        assert ars.trajectory_quality == 100.0

    def test_recovery_with_mixed_trials(self):
        sr = _make_suite_result(pass_rate=0.5)
        ars = compute_ars(sr)
        # With 50% pass rate, all cases should have both pass and fail
        assert ars.recovery > 0

    def test_custom_weights(self):
        sr = _make_suite_result(pass_rate=0.9)
        weights = {
            "accuracy": 1.0,
            "consistency": 0.0,
            "cost_efficiency": 0.0,
            "latency": 0.0,
            "trajectory_quality": 0.0,
            "recovery": 0.0,
        }
        ars = compute_ars(sr, weights=weights)
        assert ars.score == 90.0

    def test_details_populated(self):
        sr = _make_suite_result()
        ars = compute_ars(sr)
        assert "num_cases" in ars.details
        assert "avg_cost" in ars.details
        assert "total_steps" in ars.details


class TestComputeARSFromJSON:
    """Test ARS computation from JSON report."""

    def test_from_json(self):
        report = {
            "summary": {
                "overall_pass_rate": 0.85,
                "total_cost": 0.1,
                "total_duration_ms": 5000.0,
            },
            "results": [
                {"pass_rate": 0.9, "test_case": "case-1"},
                {"pass_rate": 0.8, "test_case": "case-2"},
            ],
        }
        ars = compute_ars_from_json(report)
        assert isinstance(ars, ARSBreakdown)
        assert 0 <= ars.score <= 100
        assert ars.accuracy == 85.0

    def test_empty_report(self):
        report = {"summary": {}, "results": []}
        ars = compute_ars_from_json(report)
        # cost_efficiency + latency + trajectory_quality = 3 * (0.1 * 100) = 30
        assert ars.score == 30.0

    def test_source_in_details(self):
        report = {"summary": {}, "results": []}
        ars = compute_ars_from_json(report)
        assert ars.details.get("source") == "json"
