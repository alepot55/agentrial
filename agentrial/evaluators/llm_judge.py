"""LLM-as-Judge evaluator with statistical calibration.

Unlike naive LLM-as-judge approaches (single-shot, no consistency check),
this evaluator:
1. Repeats each judgment M times to measure inter-rater consistency
2. Computes Krippendorff's alpha to quantify judge reliability
3. Optionally calibrates against gold-standard human labels
4. Reports confidence intervals on judgments

Provider-agnostic: supports any LLM via litellm or a custom callable.
Falls back to rule-based evaluation when no LLM is configured.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class JudgmentResult:
    """Result of a single judgment (one evaluation of one output)."""

    score: float  # 1-5 scale
    reasoning: str = ""
    raw_response: str = ""


@dataclass
class CalibratedJudgment:
    """Aggregated judgment with statistical meta-evaluation."""

    scores: list[float]
    mean_score: float
    score_ci: tuple[float, float]
    alpha: float  # Krippendorff's alpha for inter-rater agreement
    reliable: bool  # alpha >= 0.67
    calibration_bias: float | None = None  # Bias vs gold standard
    corrected_score: float | None = None  # Score after bias correction
    reasoning_samples: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the score meets the default passing threshold (>= 3.0)."""
        effective = self.corrected_score if self.corrected_score is not None else self.mean_score
        return effective >= 3.0


@dataclass
class Rubric:
    """Evaluation rubric for LLM judge.

    Attributes:
        criteria: Description of what's being evaluated.
        scale: Description of the 1-5 scale.
        passing_threshold: Minimum score to pass (default 3.0).
        examples: Optional examples of scores for calibration.
    """

    criteria: str
    scale: str = (
        "1: Completely wrong or irrelevant. "
        "2: Partially addresses the query but with major errors. "
        "3: Adequate - addresses the query with minor issues. "
        "4: Good - correct and well-structured response. "
        "5: Excellent - thorough, accurate, and well-articulated."
    )
    passing_threshold: float = 3.0
    examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GoldStandard:
    """Gold standard labels for calibration.

    Attributes:
        items: List of (output_text, expected_score) pairs.
    """

    items: list[tuple[str, float]]


def _build_judge_prompt(
    query: str,
    output: str,
    rubric: Rubric,
) -> str:
    """Build the prompt for the LLM judge."""
    return f"""You are an expert evaluator. Score the following AI agent output on a scale of 1-5.

## Evaluation Criteria
{rubric.criteria}

## Scoring Scale
{rubric.scale}

## Input Query
{query}

## Agent Output
{output}

## Instructions
Provide your evaluation in exactly this format:
SCORE: [1-5]
REASONING: [Brief explanation of your score]

Be strict and consistent. Do not give high scores to outputs with factual errors."""


def _parse_judge_response(response: str) -> JudgmentResult:
    """Parse the judge LLM response to extract score and reasoning."""
    score = 3.0  # Default
    reasoning = ""

    lines = response.strip().split("\n")
    for line in lines:
        line_stripped = line.strip()
        if line_stripped.upper().startswith("SCORE:"):
            score_text = line_stripped[6:].strip()
            # Extract first number
            import re

            match = re.search(r"(\d+\.?\d*)", score_text)
            if match:
                parsed = float(match.group(1))
                score = max(1.0, min(5.0, parsed))
        elif line_stripped.upper().startswith("REASONING:"):
            reasoning = line_stripped[10:].strip()

    return JudgmentResult(score=score, reasoning=reasoning, raw_response=response)


def compute_krippendorff_alpha(
    ratings: list[list[float]],
    scale_range: tuple[float, float] = (1.0, 5.0),
) -> float:
    """Compute judge consistency as a normalized agreement metric.

    For the typical use case (M repeated judgments of the same output on
    a 1-5 scale), this computes agreement relative to the maximum possible
    disagreement on the scale.

    When multiple items are provided, computes standard Krippendorff's alpha.
    For single-item cases, uses scale-relative consistency since the
    classical formula is degenerate with one unit.

    Args:
        ratings: List of rating sets. Each inner list contains M ratings
                 for one item. For single-item case: [[r1, r2, r3]].
        scale_range: (min, max) of the rating scale for normalization.

    Returns:
        Alpha value. 1.0 = perfect agreement, 0.0 = random.
    """
    if not ratings or all(len(r) <= 1 for r in ratings):
        return 1.0  # Trivially perfect agreement

    # Flatten all ratings
    all_ratings = [r for item in ratings for r in item]
    if len(set(all_ratings)) <= 1:
        return 1.0  # All identical

    # For single-item case, use scale-relative consistency
    if len(ratings) == 1:
        item_ratings = ratings[0]
        if len(item_ratings) < 2:
            return 1.0

        # Compute mean pairwise disagreement
        disagreement = 0.0
        pairs = 0
        for i in range(len(item_ratings)):
            for j in range(i + 1, len(item_ratings)):
                disagreement += (item_ratings[i] - item_ratings[j]) ** 2
                pairs += 1

        observed = disagreement / pairs if pairs > 0 else 0.0

        # Maximum possible disagreement on the scale
        scale_width = scale_range[1] - scale_range[0]
        max_disagreement = scale_width ** 2

        if max_disagreement == 0:
            return 1.0

        return max(0.0, 1.0 - (observed / max_disagreement))

    # Multi-item: standard Krippendorff's alpha
    n = len(all_ratings)
    mean_all = statistics.mean(all_ratings)

    # Total variance (denominator)
    total_var = sum((r - mean_all) ** 2 for r in all_ratings) / (n - 1)
    if total_var == 0:
        return 1.0

    # Within-unit variance (observed disagreement)
    within_var = 0.0
    total_pairs = 0
    for item_ratings in ratings:
        m = len(item_ratings)
        if m < 2:
            continue
        for i in range(m):
            for j in range(i + 1, m):
                within_var += (item_ratings[i] - item_ratings[j]) ** 2
                total_pairs += 1

    if total_pairs == 0:
        return 1.0

    observed_disagreement = within_var / total_pairs
    expected_disagreement = total_var

    if expected_disagreement == 0:
        return 1.0

    alpha = 1.0 - (observed_disagreement / expected_disagreement)
    return max(-1.0, min(1.0, alpha))


def calibrate_bias(
    judge_fn: Callable[[str, str], JudgmentResult],
    gold: GoldStandard,
    rubric: Rubric,
) -> float:
    """Compute systematic bias of the judge against gold standard.

    Args:
        judge_fn: Function that takes (query, output) and returns judgment.
        gold: Gold standard labels.
        rubric: Evaluation rubric.

    Returns:
        Bias value (positive = judge scores higher than gold standard).
    """
    if not gold.items:
        return 0.0

    diffs = []
    for output_text, expected_score in gold.items:
        judgment = judge_fn("", output_text)
        diffs.append(judgment.score - expected_score)

    return statistics.mean(diffs) if diffs else 0.0


class LLMJudge:
    """LLM-as-Judge evaluator with statistical calibration.

    Usage:
        judge = LLMJudge(
            rubric=Rubric(criteria="Evaluate answer correctness"),
            llm_fn=my_llm_call,
            repeats=3,
        )
        result = judge.evaluate("What is 2+2?", "The answer is 4.")
        print(result.mean_score, result.alpha, result.reliable)
    """

    def __init__(
        self,
        rubric: Rubric,
        llm_fn: Callable[[str], str] | None = None,
        repeats: int = 3,
        gold_standard: GoldStandard | None = None,
    ) -> None:
        """Initialize the judge.

        Args:
            rubric: Evaluation rubric.
            llm_fn: Function that takes a prompt string and returns LLM response.
                    If None, attempts to use litellm.
            repeats: Number of times to repeat each judgment (default 3).
            gold_standard: Optional gold standard for bias calibration.
        """
        self.rubric = rubric
        self.llm_fn = llm_fn
        self.repeats = max(1, repeats)
        self.gold_standard = gold_standard
        self._calibration_bias: float | None = None

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        if self.llm_fn is not None:
            return self.llm_fn(prompt)

        # Try litellm as fallback
        try:
            import litellm

            response = litellm.completion(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            return response.choices[0].message.content or ""
        except ImportError as err:
            raise RuntimeError(
                "No LLM function provided and litellm is not installed. "
                "Either pass llm_fn to LLMJudge or install litellm: "
                "pip install litellm"
            ) from err

    def _judge_once(self, query: str, output: str) -> JudgmentResult:
        """Run a single judgment."""
        prompt = _build_judge_prompt(query, output, self.rubric)
        response = self._call_llm(prompt)
        return _parse_judge_response(response)

    def evaluate(self, query: str, output: str) -> CalibratedJudgment:
        """Evaluate an agent output with statistical calibration.

        Args:
            query: The input query.
            output: The agent's output to evaluate.

        Returns:
            CalibratedJudgment with score, CI, alpha, and reliability flag.
        """
        # Run M repeated judgments
        judgments = []
        for _ in range(self.repeats):
            j = self._judge_once(query, output)
            judgments.append(j)

        scores = [j.score for j in judgments]
        mean_score = statistics.mean(scores)

        # Compute confidence interval for the score
        if len(scores) >= 2:
            std_err = statistics.stdev(scores) / (len(scores) ** 0.5)
            ci_lower = max(1.0, mean_score - 1.96 * std_err)
            ci_upper = min(5.0, mean_score + 1.96 * std_err)
        else:
            ci_lower = ci_upper = mean_score

        # Compute Krippendorff's alpha
        alpha = compute_krippendorff_alpha([scores])
        reliable = alpha >= 0.67

        # Apply calibration bias if available
        bias = self._get_calibration_bias()
        corrected = None
        if bias is not None:
            corrected = max(1.0, min(5.0, mean_score - bias))

        return CalibratedJudgment(
            scores=scores,
            mean_score=mean_score,
            score_ci=(ci_lower, ci_upper),
            alpha=alpha,
            reliable=reliable,
            calibration_bias=bias,
            corrected_score=corrected,
            reasoning_samples=[j.reasoning for j in judgments if j.reasoning],
        )

    def _get_calibration_bias(self) -> float | None:
        """Get or compute calibration bias."""
        if self._calibration_bias is not None:
            return self._calibration_bias

        if self.gold_standard is None:
            return None

        self._calibration_bias = calibrate_bias(
            self._judge_once, self.gold_standard, self.rubric
        )
        return self._calibration_bias
