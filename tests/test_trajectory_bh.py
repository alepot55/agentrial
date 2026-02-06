"""Tests for Benjamini-Hochberg correction in failure attribution."""

from __future__ import annotations

import random

from agentrial.metrics.trajectory import attribute_failures
from agentrial.types import (
    AgentInput,
    AgentMetadata,
    AgentOutput,
    StepType,
    TestCase,
    TrajectoryStep,
    TrialResult,
)


def _make_trial(
    passed: bool,
    step_names: list[str],
    trial_index: int = 0,
) -> TrialResult:
    """Create a trial with given steps."""
    steps = [
        TrajectoryStep(
            step_index=i,
            step_type=StepType.TOOL_CALL,
            name=name,
            duration_ms=50.0,
        )
        for i, name in enumerate(step_names)
    ]
    return TrialResult(
        trial_index=trial_index,
        test_case=TestCase(name="test", input=AgentInput(query="q")),
        agent_output=AgentOutput(
            output="ok" if passed else "fail",
            steps=steps,
            metadata=AgentMetadata(cost=0.01, duration_ms=100.0),
        ),
        passed=passed,
        duration_ms=100.0,
        cost=0.01,
        failures=[] if passed else ["failed"],
    )


class TestBHCorrectionInAttribution:
    """Test that BH correction prevents false positives."""

    def test_no_false_positives_with_identical_distributions(self) -> None:
        """With 10 steps drawn from the SAME distribution for success and fail,
        BH correction should control false discovery rate < 5%.

        Run multiple rounds to estimate the false positive rate.
        """
        random.seed(42)
        actions_pool = ["action_a", "action_b"]
        false_positive_count = 0
        n_rounds = 100

        for _ in range(n_rounds):
            trials = []
            # Create 20 trials (10 pass, 10 fail) with 10 steps each,
            # all drawn from the same distribution
            for t_idx in range(20):
                passed = t_idx < 10
                step_names = [random.choice(actions_pool) for _ in range(10)]
                trials.append(_make_trial(passed, step_names, trial_index=t_idx))

            result = attribute_failures(trials)

            # If any step is flagged as significant, it's a false positive
            # (since all distributions are identical)
            if result["most_likely_step"] is not None:
                # Check if the step is actually flagged as significant
                divergences = result["step_divergences"]
                sig_steps = [d for d in divergences if d.get("significant", False)]
                if sig_steps:
                    false_positive_count += 1

        # With BH correction, FDR should be < 5%
        # Allow some slack for randomness (use 15% as upper bound for 100 rounds)
        fp_rate = false_positive_count / n_rounds
        assert fp_rate < 0.15, (
            f"False positive rate {fp_rate:.0%} is too high. "
            f"BH correction should control FDR below 5%."
        )

    def test_bh_corrects_p_values(self) -> None:
        """Verify that attribute_failures returns adjusted p-values."""
        random.seed(123)
        trials = []
        for t_idx in range(20):
            passed = t_idx < 10
            step_names = [random.choice(["a", "b"]) for _ in range(5)]
            trials.append(_make_trial(passed, step_names, trial_index=t_idx))

        result = attribute_failures(trials)
        divergences = result["step_divergences"]

        # All steps should have adjusted p-values
        for div in divergences:
            assert "raw_p_value" in div, "Missing raw_p_value"
            assert "significant" in div, "Missing significant flag"
            # Adjusted p-value should be >= raw p-value
            assert div["p_value"] >= div["raw_p_value"] - 1e-10

    def test_real_divergence_detected(self) -> None:
        """When one step truly differs, BH should still detect it."""
        random.seed(456)
        trials = []
        for t_idx in range(30):
            passed = t_idx < 15
            # Steps 0-4: same distribution for both
            step_names = [random.choice(["a", "b"]) for _ in range(5)]
            # Step 5: truly different between pass and fail
            if passed:
                step_names.append("good_action")
            else:
                step_names.append("bad_action")
            trials.append(_make_trial(passed, step_names, trial_index=t_idx))

        result = attribute_failures(trials)

        # Step 5 should be flagged as the most likely failure cause
        assert result["most_likely_step"] == 5
        divergences = result["step_divergences"]
        step5 = divergences[5]
        assert step5["significant"] is True
        assert step5["p_value"] < 0.05
