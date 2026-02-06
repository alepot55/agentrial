"""Model pricing for cost calculation.

Provides pricing information for various LLM models to calculate
real costs based on token usage.
"""

import logging

logger = logging.getLogger(__name__)

# Pricing per million tokens (USD)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # --- Anthropic ---
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
    "claude-3-5-sonnet-20240620": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-5-20250929": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-opus-4-20250918": {"input": 15.0, "output": 75.0},
    # --- OpenAI ---
    "gpt-4o": {"input": 2.50, "output": 10.0},
    "gpt-4o-2024-11-20": {"input": 2.50, "output": 10.0},
    "gpt-4o-2024-08-06": {"input": 2.50, "output": 10.0},
    "gpt-4o-2024-05-13": {"input": 5.0, "output": 15.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o-mini-2024-07-18": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4-turbo-2024-04-09": {"input": 10.0, "output": 30.0},
    "gpt-4-0125-preview": {"input": 10.0, "output": 30.0},
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-3.5-turbo-0125": {"input": 0.50, "output": 1.50},
    "o1-preview": {"input": 15.0, "output": 60.0},
    "o1-mini": {"input": 3.0, "output": 12.0},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # --- Google ---
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.0},
    "gemini-1.5-pro-002": {"input": 1.25, "output": 5.0},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-1.5-flash-002": {"input": 0.075, "output": 0.30},
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    # --- Mistral ---
    "mistral-large-2411": {"input": 2.0, "output": 6.0},
    "mistral-large": {"input": 2.0, "output": 6.0},
    "mistral-medium-2312": {"input": 2.7, "output": 8.1},
    "mistral-medium": {"input": 2.7, "output": 8.1},
    "mistral-small-2409": {"input": 0.2, "output": 0.6},
    "mistral-small": {"input": 0.2, "output": 0.6},
    "mistral-nemo": {"input": 0.15, "output": 0.15},
    "codestral": {"input": 0.3, "output": 0.9},
    "pixtral-large": {"input": 2.0, "output": 6.0},
    # --- Meta (via API providers like Together, Fireworks, Groq) ---
    "llama-3.3-70b": {"input": 0.59, "output": 0.79},
    "llama-3.1-70b": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b": {"input": 0.10, "output": 0.10},
    # --- DeepSeek ---
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
}

# Default pricing for unknown models (conservative estimate)
DEFAULT_PRICING: dict[str, float] = {"input": 1.0, "output": 3.0}


def get_model_pricing(model_name: str) -> dict[str, float]:
    """Get pricing for a model.

    Args:
        model_name: The model name or identifier.

    Returns:
        Dict with 'input' and 'output' prices per million tokens.
    """
    # Exact match
    if model_name in MODEL_PRICING:
        return MODEL_PRICING[model_name]

    # Try partial matching for model families
    model_lower = model_name.lower()

    # Anthropic patterns
    if "claude-3-5-haiku" in model_lower or "claude-3-haiku" in model_lower:
        return MODEL_PRICING["claude-3-5-haiku-20241022"]
    if "claude-sonnet-4-5" in model_lower or "claude-sonnet-4.5" in model_lower:
        return MODEL_PRICING["claude-sonnet-4-5-20250929"]
    if "claude-3-5-sonnet" in model_lower or "claude-3-sonnet" in model_lower:
        return MODEL_PRICING["claude-3-5-sonnet-20241022"]
    if "claude-sonnet-4" in model_lower:
        return MODEL_PRICING["claude-sonnet-4-20250514"]
    if "claude-opus-4" in model_lower:
        return MODEL_PRICING["claude-opus-4-20250918"]
    if "claude-3-opus" in model_lower or "claude-opus" in model_lower:
        return MODEL_PRICING["claude-3-opus-20240229"]

    # OpenAI patterns
    if "gpt-4o-mini" in model_lower:
        return MODEL_PRICING["gpt-4o-mini"]
    if "gpt-4o" in model_lower:
        return MODEL_PRICING["gpt-4o"]
    if "gpt-4-turbo" in model_lower or "gpt-4-0125" in model_lower:
        return MODEL_PRICING["gpt-4-turbo"]
    if "gpt-4" in model_lower:
        return MODEL_PRICING["gpt-4"]
    if "gpt-3.5" in model_lower:
        return MODEL_PRICING["gpt-3.5-turbo"]
    if "o3-mini" in model_lower:
        return MODEL_PRICING["o3-mini"]
    if "o1-mini" in model_lower:
        return MODEL_PRICING["o1-mini"]
    if "o1-preview" in model_lower or "o1" in model_lower:
        return MODEL_PRICING["o1-preview"]

    # Gemini patterns
    if "gemini-2.0-flash-lite" in model_lower:
        return MODEL_PRICING["gemini-2.0-flash-lite"]
    if "gemini-2.0-flash" in model_lower or "gemini-2.0" in model_lower:
        return MODEL_PRICING["gemini-2.0-flash"]
    if "gemini-1.5-pro" in model_lower:
        return MODEL_PRICING["gemini-1.5-pro"]
    if "gemini-1.5-flash" in model_lower:
        return MODEL_PRICING["gemini-1.5-flash"]
    if "gemini" in model_lower:
        return MODEL_PRICING["gemini-1.0-pro"]

    # Mistral patterns
    if "codestral" in model_lower:
        return MODEL_PRICING["codestral"]
    if "pixtral" in model_lower:
        return MODEL_PRICING["pixtral-large"]
    if "mistral-nemo" in model_lower:
        return MODEL_PRICING["mistral-nemo"]
    if "mistral-large" in model_lower:
        return MODEL_PRICING["mistral-large"]
    if "mistral-medium" in model_lower:
        return MODEL_PRICING["mistral-medium"]
    if "mistral-small" in model_lower or "mistral" in model_lower:
        return MODEL_PRICING["mistral-small"]

    # Meta patterns
    if "llama-3.3" in model_lower:
        return MODEL_PRICING["llama-3.3-70b"]
    if "llama-3.1-8b" in model_lower or "llama-3.1-8" in model_lower:
        return MODEL_PRICING["llama-3.1-8b"]
    if "llama-3.1" in model_lower or "llama" in model_lower:
        return MODEL_PRICING["llama-3.1-70b"]

    # DeepSeek patterns
    if "deepseek-reasoner" in model_lower or "deepseek-r1" in model_lower:
        return MODEL_PRICING["deepseek-reasoner"]
    if "deepseek" in model_lower:
        return MODEL_PRICING["deepseek-chat"]

    # Unknown model - use default and log warning
    logger.warning(
        "Unknown model '%s', using default pricing ($%.2f/M input, $%.2f/M output)",
        model_name,
        DEFAULT_PRICING["input"],
        DEFAULT_PRICING["output"],
    )
    return DEFAULT_PRICING


def calculate_cost(
    model_name: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost for a model invocation.

    Args:
        model_name: The model name or identifier.
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of output/completion tokens.

    Returns:
        Cost in USD.
    """
    pricing = get_model_pricing(model_name)

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def estimate_cost_from_total_tokens(
    model_name: str | None,
    total_tokens: int,
    input_ratio: float = 0.5,
) -> float:
    """Estimate cost when only total tokens are known.

    Args:
        model_name: The model name (or None for default).
        total_tokens: Total token count.
        input_ratio: Estimated ratio of input to total tokens (default 0.5).

    Returns:
        Estimated cost in USD.
    """
    if model_name:
        pricing = get_model_pricing(model_name)
    else:
        pricing = DEFAULT_PRICING

    input_tokens = int(total_tokens * input_ratio)
    output_tokens = total_tokens - input_tokens

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost
