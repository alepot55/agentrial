"""Tests for LLM-as-Judge evaluator."""

from agentrial.evaluators.llm_judge import (
    CalibratedJudgment,
    GoldStandard,
    LLMJudge,
    Rubric,
    _parse_judge_response,
    _t_ppf_975,
    compute_krippendorff_alpha,
)


class TestParseJudgeResponse:
    """Tests for parsing LLM judge responses."""

    def test_standard_format(self) -> None:
        response = "SCORE: 4\nREASONING: Good answer with minor issues."
        result = _parse_judge_response(response)
        assert result.score == 4.0
        assert "Good answer" in result.reasoning

    def test_score_with_decimal(self) -> None:
        response = "SCORE: 3.5\nREASONING: Acceptable."
        result = _parse_judge_response(response)
        assert result.score == 3.5

    def test_score_clamped_high(self) -> None:
        response = "SCORE: 10\nREASONING: Incredible."
        result = _parse_judge_response(response)
        assert result.score == 5.0

    def test_score_clamped_low(self) -> None:
        response = "SCORE: 0\nREASONING: Terrible."
        result = _parse_judge_response(response)
        assert result.score == 1.0

    def test_malformed_response(self) -> None:
        response = "I think this is a 4 out of 5."
        result = _parse_judge_response(response)
        # Should use default score
        assert result.score == 3.0

    def test_extra_whitespace(self) -> None:
        response = "  SCORE:  4  \n  REASONING:  Good.  "
        result = _parse_judge_response(response)
        assert result.score == 4.0


class TestKrippendorffAlpha:
    """Tests for Krippendorff's alpha computation."""

    def test_perfect_agreement(self) -> None:
        # All raters give the same score
        ratings = [[4.0, 4.0, 4.0]]
        alpha = compute_krippendorff_alpha(ratings)
        assert alpha == 1.0

    def test_high_agreement(self) -> None:
        # Raters mostly agree (4, 4, 5 on 1-5 scale)
        ratings = [[4.0, 4.0, 5.0]]
        alpha = compute_krippendorff_alpha(ratings)
        # Disagreement is (1^2)/3 = 0.33, max is 16, so alpha = 1 - 0.33/16 ≈ 0.98
        assert alpha > 0.9

    def test_low_agreement(self) -> None:
        # Raters disagree significantly (1 and 5 on 1-5 scale)
        ratings = [[1.0, 5.0, 1.0, 5.0]]
        alpha = compute_krippendorff_alpha(ratings)
        assert alpha < 0.9

    def test_single_rating(self) -> None:
        # Single rating => trivially perfect
        ratings = [[4.0]]
        alpha = compute_krippendorff_alpha(ratings)
        assert alpha == 1.0

    def test_empty_ratings(self) -> None:
        alpha = compute_krippendorff_alpha([])
        assert alpha == 1.0

    def test_multiple_items(self) -> None:
        # Two items, each with 3 ratings
        ratings = [
            [4.0, 4.0, 5.0],  # High agreement
            [2.0, 3.0, 2.0],  # Moderate agreement
        ]
        alpha = compute_krippendorff_alpha(ratings)
        assert -1.0 <= alpha <= 1.0


class TestLLMJudge:
    """Tests for the LLMJudge class."""

    def _make_mock_llm(self, score: float = 4.0) -> callable:
        """Create a mock LLM that returns a fixed score."""

        def mock_llm(prompt: str) -> str:
            return f"SCORE: {score}\nREASONING: Mock evaluation."

        return mock_llm

    def test_basic_evaluation(self) -> None:
        rubric = Rubric(criteria="Is the answer correct?")
        judge = LLMJudge(
            rubric=rubric,
            llm_fn=self._make_mock_llm(4.0),
            repeats=3,
        )

        result = judge.evaluate("What is 2+2?", "The answer is 4.")
        assert result.mean_score == 4.0
        assert result.alpha == 1.0  # All identical scores
        assert result.reliable
        assert result.passed

    def test_low_score_fails(self) -> None:
        rubric = Rubric(criteria="Is the answer correct?")
        judge = LLMJudge(
            rubric=rubric,
            llm_fn=self._make_mock_llm(2.0),
            repeats=3,
        )

        result = judge.evaluate("What is 2+2?", "The answer is 5.")
        assert result.mean_score == 2.0
        assert not result.passed

    def test_variable_scores(self) -> None:
        """Test with varying scores to check CI and alpha."""
        call_count = 0

        def varying_llm(prompt: str) -> str:
            nonlocal call_count
            scores = [3.0, 4.0, 5.0]
            score = scores[call_count % len(scores)]
            call_count += 1
            return f"SCORE: {score}\nREASONING: Evaluation {call_count}."

        rubric = Rubric(criteria="Quality check")
        judge = LLMJudge(
            rubric=rubric,
            llm_fn=varying_llm,
            repeats=3,
        )

        result = judge.evaluate("test", "test output")
        assert result.mean_score == 4.0  # mean of [3, 4, 5]
        assert len(result.scores) == 3
        assert result.score_ci[0] < result.mean_score
        assert result.score_ci[1] > result.mean_score

    def test_calibration_with_gold_standard(self) -> None:
        """Test bias correction with gold standard labels."""
        # Mock LLM that consistently scores 0.5 higher than gold
        def biased_llm(prompt: str) -> str:
            return "SCORE: 4.5\nREASONING: Looks good."

        gold = GoldStandard(items=[
            ("Good answer about Paris", 4.0),
            ("Another good answer", 4.0),
        ])

        rubric = Rubric(criteria="Correctness")
        judge = LLMJudge(
            rubric=rubric,
            llm_fn=biased_llm,
            repeats=1,
            gold_standard=gold,
        )

        result = judge.evaluate("What is the capital?", "Paris is the capital.")
        assert result.calibration_bias is not None
        assert result.calibration_bias > 0  # Judge scores higher
        assert result.corrected_score is not None
        assert result.corrected_score < result.mean_score  # Corrected downward

    def test_single_repeat(self) -> None:
        rubric = Rubric(criteria="Check")
        judge = LLMJudge(
            rubric=rubric,
            llm_fn=self._make_mock_llm(3.5),
            repeats=1,
        )

        result = judge.evaluate("q", "a")
        assert result.mean_score == 3.5
        assert result.alpha == 1.0

    def test_no_llm_falls_back_to_rule_based(self) -> None:
        """Test that missing LLM function falls back to rule-based judge."""
        rubric = Rubric(criteria="Check")
        judge = LLMJudge(rubric=rubric, llm_fn=None, repeats=1)

        result = judge.evaluate("q", "a")
        assert judge._rule_based is True
        assert result.mean_score >= 1.0
        assert result.mean_score <= 5.0


class TestCalibratedJudgment:
    """Tests for CalibratedJudgment properties."""

    def test_passed_with_high_score(self) -> None:
        j = CalibratedJudgment(
            scores=[4.0], mean_score=4.0, score_ci=(3.5, 4.5),
            alpha=1.0, reliable=True,
        )
        assert j.passed

    def test_failed_with_low_score(self) -> None:
        j = CalibratedJudgment(
            scores=[2.0], mean_score=2.0, score_ci=(1.5, 2.5),
            alpha=1.0, reliable=True,
        )
        assert not j.passed

    def test_passed_uses_corrected_score(self) -> None:
        j = CalibratedJudgment(
            scores=[3.5], mean_score=3.5, score_ci=(3.0, 4.0),
            alpha=1.0, reliable=True,
            corrected_score=2.5,  # After bias correction, below threshold
        )
        assert not j.passed

    def test_reasoning_samples(self) -> None:
        j = CalibratedJudgment(
            scores=[4.0], mean_score=4.0, score_ci=(4.0, 4.0),
            alpha=1.0, reliable=True,
            reasoning_samples=["Good", "Correct"],
        )
        assert len(j.reasoning_samples) == 2


class TestTDistribution:
    """Tests for t-distribution CI (H1 fix)."""

    def test_t_ppf_975_known_values(self) -> None:
        assert _t_ppf_975(1) == 12.706
        assert _t_ppf_975(2) == 4.303
        assert _t_ppf_975(30) == 2.042
        assert _t_ppf_975(120) == 1.980

    def test_t_ppf_975_large_df_fallback(self) -> None:
        # df > 120 should use 1.96 fallback
        assert _t_ppf_975(500) == 1.96

    def test_ci_uses_t_distribution(self) -> None:
        """With M=3, CI should be wider than z=1.96 would give."""
        call_count = 0

        def varying_llm(prompt: str) -> str:
            nonlocal call_count
            scores = [3.0, 4.0, 5.0]
            score = scores[call_count % len(scores)]
            call_count += 1
            return f"SCORE: {score}\nREASONING: Test."

        rubric = Rubric(criteria="Test")
        judge = LLMJudge(rubric=rubric, llm_fn=varying_llm, repeats=3)
        result = judge.evaluate("q", "a")

        # With t(df=2)=4.303 vs z=1.96, CI should be wider
        # stdev([3,4,5]) = 1.0, std_err = 1/sqrt(3) ≈ 0.577
        # z-CI width: 2 * 1.96 * 0.577 ≈ 2.26
        # t-CI width: 2 * 4.303 * 0.577 ≈ 4.97
        ci_width = result.score_ci[1] - result.score_ci[0]
        # t-CI should be wider than z-CI would be
        z_ci_width = 2 * 1.96 * 1.0 / (3 ** 0.5)
        assert ci_width > z_ci_width

    def test_ci_warning_for_small_m(self) -> None:
        """Verify warning is logged when M < 5."""
        import logging

        rubric = Rubric(criteria="Test")
        judge = LLMJudge(
            rubric=rubric,
            llm_fn=lambda p: "SCORE: 4\nREASONING: ok",
            repeats=3,
        )

        logger = logging.getLogger("agentrial.evaluators.llm_judge")
        with unittest_mock_handler(logger) as handler:
            result = judge.evaluate("q", "a")
            assert result.mean_score == 4.0
            # Check that a warning was logged about small M
            warnings = [
                r for r in handler.records if r.levelno == logging.WARNING
            ]
            assert any("only 3 repeats" in r.getMessage() for r in warnings)


def unittest_mock_handler(lgr: "logging.Logger"):
    """Context manager that attaches a handler to capture log records."""
    import contextlib
    import logging

    @contextlib.contextmanager
    def _ctx():
        handler = logging.Handler()
        handler.records: list[logging.LogRecord] = []  # type: ignore[attr-defined]
        original_emit = handler.emit
        handler.emit = lambda record: handler.records.append(record)  # type: ignore[assignment]
        lgr.addHandler(handler)
        try:
            yield handler
        finally:
            lgr.removeHandler(handler)

    return _ctx()
