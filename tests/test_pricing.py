"""Tests for model pricing and cost calculation."""

import logging

from agentrial.runner.adapters.pricing import (
    MODEL_PRICING,
    calculate_cost,
    estimate_cost_from_total_tokens,
    get_model_pricing,
)


class TestModelPricingDict:
    """Tests for the MODEL_PRICING dictionary."""

    def test_has_at_least_40_models(self):
        assert len(MODEL_PRICING) >= 40, (
            f"MODEL_PRICING has {len(MODEL_PRICING)} models, expected >= 40"
        )

    def test_all_entries_have_input_and_output(self):
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing, f"{model} missing 'input' price"
            assert "output" in pricing, f"{model} missing 'output' price"
            assert pricing["input"] >= 0, f"{model} has negative input price"
            assert pricing["output"] >= 0, f"{model} has negative output price"

    def test_has_anthropic_models(self):
        assert any("claude" in k for k in MODEL_PRICING)

    def test_has_openai_models(self):
        assert any("gpt" in k for k in MODEL_PRICING)

    def test_has_google_models(self):
        assert any("gemini" in k for k in MODEL_PRICING)

    def test_has_mistral_models(self):
        assert any("mistral" in k for k in MODEL_PRICING)

    def test_has_meta_models(self):
        assert any("llama" in k for k in MODEL_PRICING)

    def test_has_deepseek_models(self):
        assert any("deepseek" in k for k in MODEL_PRICING)


class TestCalculateCost:
    """Tests for calculate_cost()."""

    def test_gpt4o_cost(self):
        # 1000 input + 500 output tokens with gpt-4o
        # Input: 1000/1M * $2.50 = $0.0025
        # Output: 500/1M * $10.0 = $0.005
        cost = calculate_cost("gpt-4o", 1000, 500)
        assert abs(cost - 0.0075) < 1e-6

    def test_claude_sonnet_cost(self):
        # 10000 input + 2000 output with claude-3-5-sonnet
        # Input: 10000/1M * $3.0 = $0.03
        # Output: 2000/1M * $15.0 = $0.03
        cost = calculate_cost("claude-3-5-sonnet-20241022", 10000, 2000)
        assert abs(cost - 0.06) < 1e-6

    def test_deepseek_chat_cost(self):
        cost = calculate_cost("deepseek-chat", 1_000_000, 1_000_000)
        assert abs(cost - (0.27 + 1.10)) < 1e-6

    def test_zero_tokens(self):
        cost = calculate_cost("gpt-4o", 0, 0)
        assert cost == 0.0


class TestGetModelPricing:
    """Tests for get_model_pricing() pattern matching."""

    def test_exact_match(self):
        pricing = get_model_pricing("gpt-4o")
        assert pricing["input"] == 2.50

    def test_partial_match_claude(self):
        pricing = get_model_pricing("claude-3-5-sonnet-latest")
        assert pricing["input"] == 3.0

    def test_partial_match_gemini_2(self):
        pricing = get_model_pricing("gemini-2.0-flash-exp")
        assert pricing["input"] == 0.10

    def test_partial_match_llama(self):
        pricing = get_model_pricing("llama-3.3-70b-instruct")
        assert pricing["input"] == 0.59

    def test_partial_match_deepseek_r1(self):
        pricing = get_model_pricing("deepseek-r1-distill")
        assert pricing["input"] == 0.55

    def test_unknown_model_returns_default(self, caplog):
        with caplog.at_level(logging.WARNING):
            pricing = get_model_pricing("totally-unknown-model-xyz")
        assert pricing["input"] == 1.0
        assert pricing["output"] == 3.0
        assert "Unknown model" in caplog.text


class TestEstimateCost:
    """Tests for estimate_cost_from_total_tokens()."""

    def test_with_model(self):
        cost = estimate_cost_from_total_tokens("gpt-4o-mini", 1000)
        # 500 input * 0.15/1M + 500 output * 0.60/1M
        expected = (500 / 1_000_000 * 0.15) + (500 / 1_000_000 * 0.60)
        assert abs(cost - expected) < 1e-9

    def test_without_model(self):
        cost = estimate_cost_from_total_tokens(None, 1000)
        # Uses default pricing
        expected = (500 / 1_000_000 * 1.0) + (500 / 1_000_000 * 3.0)
        assert abs(cost - expected) < 1e-9

    def test_custom_ratio(self):
        cost = estimate_cost_from_total_tokens("gpt-4o-mini", 1000, input_ratio=0.8)
        expected = (800 / 1_000_000 * 0.15) + (200 / 1_000_000 * 0.60)
        assert abs(cost - expected) < 1e-9
