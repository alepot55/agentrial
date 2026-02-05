"""Tests for Cost-Accuracy Pareto Frontier."""

from __future__ import annotations

from agentrial.pareto import (
    ModelPoint,
    ParetoResult,
    analyze_from_results,
    compute_pareto_frontier,
    generate_recommendation,
    render_pareto_ascii,
)


def _point(
    name: str, rate: float, cost: float, latency: float = 100.0
) -> ModelPoint:
    """Create a ModelPoint for testing."""
    return ModelPoint(
        model_name=name,
        pass_rate=rate,
        mean_cost=cost,
        mean_latency_ms=latency,
    )


class TestComputeParetoFrontier:
    """Tests for Pareto frontier computation."""

    def test_empty_points(self) -> None:
        assert compute_pareto_frontier([]) == []

    def test_single_point(self) -> None:
        p = _point("model-a", 0.9, 0.10)
        frontier = compute_pareto_frontier([p])
        assert len(frontier) == 1
        assert frontier[0].pareto_optimal

    def test_dominated_point(self) -> None:
        p1 = _point("better", 0.9, 0.05)
        p2 = _point("worse", 0.8, 0.10)  # Worse in both dimensions
        frontier = compute_pareto_frontier([p1, p2])

        assert len(frontier) == 1
        assert frontier[0].model_name == "better"
        assert p2.dominated

    def test_pareto_frontier_tradeoff(self) -> None:
        # These trade off: one is cheaper, other is better quality
        p1 = _point("cheap", 0.7, 0.02)
        p2 = _point("quality", 0.95, 0.15)
        frontier = compute_pareto_frontier([p1, p2])

        assert len(frontier) == 2
        assert all(p.pareto_optimal for p in frontier)

    def test_classic_scenario(self) -> None:
        points = [
            _point("gpt-4o", 0.82, 0.15),
            _point("claude-sonnet", 0.85, 0.12),  # Dominates gpt-4o
            _point("gpt-4o-mini", 0.71, 0.03),
            _point("claude-haiku", 0.68, 0.02),
        ]
        frontier = compute_pareto_frontier(points)

        # claude-sonnet dominates gpt-4o (better rate, lower cost)
        gpt4o = next(p for p in points if p.model_name == "gpt-4o")
        assert gpt4o.dominated

        # claude-sonnet, gpt-4o-mini, claude-haiku should be on frontier
        frontier_names = {p.model_name for p in frontier}
        assert "claude-sonnet" in frontier_names
        # gpt-4o-mini and claude-haiku trade off (haiku cheaper but lower rate)
        assert "claude-haiku" in frontier_names

    def test_identical_points(self) -> None:
        p1 = _point("model-a", 0.8, 0.10)
        p2 = _point("model-b", 0.8, 0.10)
        frontier = compute_pareto_frontier([p1, p2])

        # Neither dominates the other (identical metrics)
        assert len(frontier) == 2

    def test_frontier_sorted_by_cost(self) -> None:
        points = [
            _point("expensive", 0.95, 0.20),
            _point("cheap", 0.60, 0.01),
            _point("mid", 0.80, 0.08),
        ]
        frontier = compute_pareto_frontier(points)

        costs = [p.mean_cost for p in frontier]
        assert costs == sorted(costs)


class TestGenerateRecommendation:
    """Tests for recommendation generation."""

    def test_empty_frontier(self) -> None:
        result = ParetoResult()
        assert "No models" in generate_recommendation(result)

    def test_single_model(self) -> None:
        p = _point("model-a", 0.9, 0.10)
        p.pareto_optimal = True
        result = ParetoResult(
            points=[p], pareto_frontier=[p], dominated=[]
        )
        rec = generate_recommendation(result)
        assert "model-a" in rec
        assert "only Pareto-optimal" in rec

    def test_multi_model_recommendation(self) -> None:
        cheap = _point("cheap", 0.7, 0.02)
        quality = _point("quality", 0.95, 0.15)
        cheap.pareto_optimal = True
        quality.pareto_optimal = True

        dominated = _point("bad", 0.6, 0.20)
        dominated.dominated = True

        result = ParetoResult(
            points=[cheap, quality, dominated],
            pareto_frontier=[cheap, quality],
            dominated=[dominated],
        )
        rec = generate_recommendation(result)
        assert "quality" in rec.lower() or "cheap" in rec.lower()


class TestAnalyzeFromResults:
    """Tests for analyze_from_results."""

    def test_basic_analysis(self) -> None:
        model_results = {
            "gpt-4o": {
                "pass_rate": 0.82,
                "mean_cost": 0.15,
                "mean_latency_ms": 3200,
            },
            "claude-sonnet": {
                "pass_rate": 0.85,
                "mean_cost": 0.12,
                "mean_latency_ms": 2800,
            },
            "gpt-4o-mini": {
                "pass_rate": 0.71,
                "mean_cost": 0.03,
                "mean_latency_ms": 1200,
            },
        }
        result = analyze_from_results(model_results)

        assert len(result.points) == 3
        assert len(result.pareto_frontier) >= 1
        assert result.recommendation != ""
        assert result.best_quality is not None
        assert result.best_value is not None

    def test_all_pareto_optimal(self) -> None:
        # Each trades off: better accuracy = higher cost
        model_results = {
            "cheap": {"pass_rate": 0.5, "mean_cost": 0.01},
            "mid": {"pass_rate": 0.75, "mean_cost": 0.05},
            "premium": {"pass_rate": 0.95, "mean_cost": 0.20},
        }
        result = analyze_from_results(model_results)

        assert len(result.pareto_frontier) == 3
        assert len(result.dominated) == 0

    def test_single_model(self) -> None:
        model_results = {
            "only": {"pass_rate": 0.8, "mean_cost": 0.10},
        }
        result = analyze_from_results(model_results)

        assert len(result.pareto_frontier) == 1
        assert result.best_quality.model_name == "only"
        assert result.best_value.model_name == "only"


class TestRenderParetoAscii:
    """Tests for ASCII rendering."""

    def test_empty_result(self) -> None:
        result = ParetoResult()
        output = render_pareto_ascii(result)
        assert "No data" in output

    def test_renders_points(self) -> None:
        p1 = _point("model-a", 0.9, 0.10)
        p1.pareto_optimal = True
        p2 = _point("model-b", 0.7, 0.05)
        p2.pareto_optimal = True

        result = ParetoResult(
            points=[p1, p2],
            pareto_frontier=[p1, p2],
        )
        output = render_pareto_ascii(result)

        assert "Pareto Frontier" in output
        assert "model-a" in output
        assert "model-b" in output
        assert "*" in output  # Pareto-optimal marker

    def test_shows_dominated_marker(self) -> None:
        p1 = _point("good", 0.9, 0.05)
        p1.pareto_optimal = True
        p2 = _point("bad", 0.7, 0.10)
        p2.dominated = True

        result = ParetoResult(
            points=[p1, p2],
            pareto_frontier=[p1],
            dominated=[p2],
        )
        output = render_pareto_ascii(result)
        assert "x" in output  # Dominated marker


class TestParetoResult:
    """Tests for ParetoResult properties."""

    def test_best_quality(self) -> None:
        p1 = _point("a", 0.95, 0.20)
        p2 = _point("b", 0.80, 0.05)
        result = ParetoResult(pareto_frontier=[p1, p2])
        assert result.best_quality.model_name == "a"

    def test_best_value(self) -> None:
        p1 = _point("a", 0.95, 0.20)
        p2 = _point("b", 0.80, 0.05)
        result = ParetoResult(pareto_frontier=[p1, p2])
        assert result.best_value.model_name == "b"

    def test_empty_frontier(self) -> None:
        result = ParetoResult()
        assert result.best_quality is None
        assert result.best_value is None
