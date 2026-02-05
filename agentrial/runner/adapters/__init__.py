"""Framework adapters for Agentrial."""

from agentrial.runner.adapters.base import BaseAdapter
from agentrial.runner.adapters.langgraph import wrap_langgraph_agent
from agentrial.runner.adapters.pricing import MODEL_PRICING, calculate_cost, get_model_pricing

# Framework adapters are lazy-imported via their wrap_* functions.
# Each adapter imports its framework only at runtime, keeping the
# core package dependency-free.

__all__ = [
    "BaseAdapter",
    # LangGraph (built-in)
    "wrap_langgraph_agent",
    # CrewAI
    "wrap_crewai_agent",
    # AutoGen
    "wrap_autogen_agent",
    # Pydantic AI
    "wrap_pydantic_ai_agent",
    # OpenAI Agents SDK
    "wrap_openai_agent",
    # smolagents (Hugging Face)
    "wrap_smolagent",
    # Pricing
    "MODEL_PRICING",
    "calculate_cost",
    "get_model_pricing",
]


def __getattr__(name: str):  # noqa: ANN001
    """Lazy import adapter wrap functions to avoid importing frameworks at module load."""
    _adapter_map = {
        "wrap_crewai_agent": "agentrial.runner.adapters.crewai",
        "wrap_autogen_agent": "agentrial.runner.adapters.autogen",
        "wrap_pydantic_ai_agent": "agentrial.runner.adapters.pydantic_ai",
        "wrap_openai_agent": "agentrial.runner.adapters.openai_agents",
        "wrap_smolagent": "agentrial.runner.adapters.smolagents",
    }

    if name in _adapter_map:
        import importlib

        module = importlib.import_module(_adapter_map[name])
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
