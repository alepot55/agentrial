"""Tests for basic metrics."""

import pytest

from agenteval.metrics.basic import (
    compute_basic_metrics,
    compute_cost_per_correct,
    compute_latency_percentiles,
    compute_token_efficiency,
)
from agenteval.types import (
    AgentMetadata,
    AgentOutput,
    TestCase,
    AgentInput,
    TrialResult,
)


def create_trial(
    passed: bool,
    cost: float = 0.01,
    duration_ms: float = 100.0,
    tokens: int = 50,
) -> TrialResult:
    """Create a test trial result."""
    return TrialResult(
        trial_index=0,
        test_case=TestCase(name="test", input=AgentInput(query="test")),
        agent_output=AgentOutput(
            output="test",
            metadata=AgentMetadata(total_tokens=tokens, cost=cost, duration_ms=duration_ms),
        ),
        passed=passed,
        duration_ms=duration_ms,
        cost=cost,
        tokens=tokens,
    )


class TestComputeBasicMetrics:
    """Tests for compute_basic_metrics."""

    def test_all_passed(self) -> None:
        """Test metrics when all trials pass."""
        trials = [create_trial(True) for _ in range(5)]
        metrics = compute_basic_metrics(trials)

        assert metrics["pass_count"] == 5
        assert metrics["fail_count"] == 0
        assert metrics["pass_rate"] == 1.0

    def test_all_failed(self) -> None:
        """Test metrics when all trials fail."""
        trials = [create_trial(False) for _ in range(5)]
        metrics = compute_basic_metrics(trials)

        assert metrics["pass_count"] == 0
        assert metrics["fail_count"] == 5
        assert metrics["pass_rate"] == 0.0

    def test_mixed_results(self) -> None:
        """Test metrics with mixed results."""
        trials = [
            create_trial(True, cost=0.01, duration_ms=100),
            create_trial(True, cost=0.02, duration_ms=200),
            create_trial(False, cost=0.03, duration_ms=300),
        ]
        metrics = compute_basic_metrics(trials)

        assert metrics["pass_count"] == 2
        assert metrics["fail_count"] == 1
        assert metrics["pass_rate"] == pytest.approx(2 / 3)
        assert metrics["mean_cost"] == pytest.approx(0.02)
        assert metrics["mean_latency_ms"] == pytest.approx(200.0)

    def test_empty_trials(self) -> None:
        """Test metrics with no trials."""
        metrics = compute_basic_metrics([])

        assert metrics["pass_count"] == 0
        assert metrics["pass_rate"] == 0.0
        assert metrics["mean_cost"] == 0.0


class TestCostPerCorrect:
    """Tests for cost per correct answer."""

    def test_all_correct(self) -> None:
        """Test cost per correct when all pass."""
        trials = [
            create_trial(True, cost=0.01),
            create_trial(True, cost=0.02),
        ]
        cpc = compute_cost_per_correct(trials)
        # Total cost 0.03, 2 correct -> 0.015 per correct
        assert cpc == pytest.approx(0.015)

    def test_some_failures(self) -> None:
        """Test cost per correct with failures."""
        trials = [
            create_trial(True, cost=0.01),
            create_trial(False, cost=0.02),  # This cost counts but doesn't add correct
        ]
        cpc = compute_cost_per_correct(trials)
        # Total cost 0.03, 1 correct -> 0.03 per correct
        assert cpc == pytest.approx(0.03)

    def test_no_correct(self) -> None:
        """Test cost per correct when none pass."""
        trials = [create_trial(False, cost=0.01)]
        cpc = compute_cost_per_correct(trials)
        assert cpc == float("inf")


class TestLatencyPercentiles:
    """Tests for latency percentiles."""

    def test_basic_percentiles(self) -> None:
        """Test computing basic percentiles."""
        trials = [
            create_trial(True, duration_ms=i * 100) for i in range(1, 11)
        ]  # 100ms to 1000ms
        percentiles = compute_latency_percentiles(trials)

        assert "p50" in percentiles
        assert "p90" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles
        assert percentiles["p50"] == pytest.approx(550.0)

    def test_empty_trials(self) -> None:
        """Test percentiles with no trials."""
        percentiles = compute_latency_percentiles([])
        assert percentiles["p50"] == 0.0


class TestTokenEfficiency:
    """Tests for token efficiency metrics."""

    def test_basic_efficiency(self) -> None:
        """Test token efficiency calculation."""
        trials = [
            create_trial(True, tokens=100),
            create_trial(True, tokens=200),
        ]
        eff = compute_token_efficiency(trials)

        assert eff["tokens_per_trial"] == 150.0
        assert eff["tokens_per_correct"] == 150.0

    def test_with_failures(self) -> None:
        """Test token efficiency with failures."""
        trials = [
            create_trial(True, tokens=100),
            create_trial(False, tokens=200),
        ]
        eff = compute_token_efficiency(trials)

        assert eff["tokens_per_trial"] == 150.0
        # Only 1 correct, total tokens 300
        assert eff["tokens_per_correct"] == 300.0

    def test_no_correct(self) -> None:
        """Test token efficiency when none correct."""
        trials = [create_trial(False, tokens=100)]
        eff = compute_token_efficiency(trials)
        assert eff["tokens_per_correct"] == float("inf")
