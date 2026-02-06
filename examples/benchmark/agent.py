"""ReAct-style benchmark agent with tool calling.

Uses raw Anthropic or Google GenAI API calls (no frameworks).
Supports both providers via AGENT_PROVIDER and AGENT_MODEL env vars.

Usage:
    AGENT_PROVIDER=anthropic AGENT_MODEL=claude-3-5-haiku-20241022 agentrial run ...
    AGENT_PROVIDER=google AGENT_MODEL=gemini-2.0-flash agentrial run ...
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

from agentrial.runner.adapters.pricing import calculate_cost
from agentrial.types import AgentInput, AgentMetadata, AgentOutput, StepType, TrajectoryStep

# ---------------------------------------------------------------------------
# Tools — deterministic, no external API calls
# ---------------------------------------------------------------------------

TOOLS_REGISTRY: dict[str, Any] = {}


def _tool(fn: Any) -> Any:
    TOOLS_REGISTRY[fn.__name__] = fn
    return fn


@_tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, parentheses, and common math functions (sqrt, sin, cos, ceil, floor, round)."""
    allowed = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "ceil": math.ceil,
        "floor": math.floor,
        "round": round,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


@_tool
def search_knowledge(topic: str) -> str:
    """Look up factual knowledge about a topic. Returns known facts."""
    knowledge = {
        "south korea": "South Korea (officially the Republic of Korea) has its capital at Seoul. Population: ~51.7 million. Currency: South Korean Won (KRW).",
        "japan": "Japan is an island country in East Asia. Capital: Tokyo. Population: ~125 million. Currency: Japanese Yen (JPY).",
        "france": "France is a country in Western Europe. Capital: Paris. Population: ~67 million. Currency: Euro (EUR).",
        "water boiling point": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard atmospheric pressure (1 atm).",
        "sky color": "The sky appears blue on a clear day due to Rayleigh scattering of sunlight by the atmosphere.",
        "flour density": "All-purpose flour has a density of approximately 125 grams per cup (US measuring cup).",
        "sales tax": "Sales tax is a consumption tax imposed by the government on the sale of goods and services. It is calculated as a percentage of the sale price.",
        "temperature conversion": "To convert Fahrenheit to Celsius: C = (F - 32) * 5/9. To convert Celsius to Fahrenheit: F = C * 9/5 + 32.",
    }
    topic_lower = topic.lower()
    for key, value in knowledge.items():
        if key in topic_lower or topic_lower in key:
            return value
    return f"No specific information found for '{topic}'. Try rephrasing your query."


@_tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between units. Supports temperature (C/F/K), length (m/km/mi/ft), weight (kg/lb/g/oz), and volume (L/gal/cup/ml)."""
    conversions: dict[tuple[str, str], Any] = {
        # Temperature
        ("f", "c"): lambda v: (v - 32) * 5 / 9,
        ("c", "f"): lambda v: v * 9 / 5 + 32,
        ("c", "k"): lambda v: v + 273.15,
        ("k", "c"): lambda v: v - 273.15,
        ("f", "k"): lambda v: (v - 32) * 5 / 9 + 273.15,
        ("k", "f"): lambda v: (v - 273.15) * 9 / 5 + 32,
        # Length
        ("m", "km"): lambda v: v / 1000,
        ("km", "m"): lambda v: v * 1000,
        ("m", "ft"): lambda v: v * 3.28084,
        ("ft", "m"): lambda v: v / 3.28084,
        ("km", "mi"): lambda v: v * 0.621371,
        ("mi", "km"): lambda v: v / 0.621371,
        # Weight
        ("kg", "lb"): lambda v: v * 2.20462,
        ("lb", "kg"): lambda v: v / 2.20462,
        ("kg", "g"): lambda v: v * 1000,
        ("g", "kg"): lambda v: v / 1000,
        ("lb", "oz"): lambda v: v * 16,
        ("oz", "lb"): lambda v: v / 16,
        # Volume
        ("l", "gal"): lambda v: v * 0.264172,
        ("gal", "l"): lambda v: v / 0.264172,
        ("l", "ml"): lambda v: v * 1000,
        ("ml", "l"): lambda v: v / 1000,
        ("cup", "ml"): lambda v: v * 236.588,
        ("ml", "cup"): lambda v: v / 236.588,
        ("cup", "g"): lambda v: v * 125,  # for flour
        ("g", "cup"): lambda v: v / 125,
    }
    key = (from_unit.lower().strip(), to_unit.lower().strip())
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4f} {to_unit}"
    return f"Cannot convert from {from_unit} to {to_unit}. Supported pairs: temperature (C/F/K), length (m/km/mi/ft), weight (kg/lb/g/oz), volume (L/gal/cup/ml)."


@_tool
def date_info(query: str) -> str:
    """Get date-related information. Supports: days_in_year, days_in_month, is_leap_year, days_between."""
    query_lower = query.lower()
    if "days" in query_lower and "year" in query_lower:
        # Extract year if present
        import re

        years = re.findall(r"\d{4}", query)
        if years:
            total = 0
            for y in years:
                yr = int(y)
                total += 366 if (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)) else 365
            if len(years) == 1:
                yr = int(years[0])
                days = 366 if (yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)) else 365
                return f"The year {yr} has {days} days."
            return f"Years {', '.join(years)} have a combined {total} days."
        # Generic "3 years"
        nums = re.findall(r"\d+", query)
        if nums:
            n = int(nums[0])
            # Assume regular years (365 each) unless specific years given
            return f"{n} regular years = {n * 365} days. Note: leap years have 366 days."
    if "leap" in query_lower:
        import re

        years = re.findall(r"\d{4}", query)
        results = []
        for y in years:
            yr = int(y)
            is_leap = yr % 4 == 0 and (yr % 100 != 0 or yr % 400 == 0)
            results.append(f"{yr}: {'leap year' if is_leap else 'not a leap year'}")
        return "; ".join(results) if results else "Please provide a year to check."
    return f"Date query not understood: '{query}'. Try 'days in year 2024' or 'is 2024 a leap year'."


# ---------------------------------------------------------------------------
# Tool schemas for LLM APIs
# ---------------------------------------------------------------------------

TOOL_SCHEMAS_ANTHROPIC = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, parentheses, sqrt, round, abs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '(15 * 23.50) * 1.085'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "Look up factual knowledge about a topic such as countries, science facts, or general knowledge.",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to look up, e.g. 'South Korea' or 'water boiling point'",
                }
            },
            "required": ["topic"],
        },
    },
    {
        "name": "unit_converter",
        "description": "Convert between units. Supports temperature (C/F/K), length (m/km/mi/ft), weight (kg/lb/g/oz), volume (L/gal/cup/ml).",
        "input_schema": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The numeric value to convert"},
                "from_unit": {"type": "string", "description": "Source unit (e.g. 'F', 'kg', 'cup')"},
                "to_unit": {"type": "string", "description": "Target unit (e.g. 'C', 'lb', 'ml')"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "date_info",
        "description": "Get date-related information such as days in a year, leap year checks, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The date question, e.g. 'days in 3 years' or 'is 2024 a leap year'",
                }
            },
            "required": ["query"],
        },
    },
]

TOOL_SCHEMAS_GOOGLE = [
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Supports +, -, *, /, **, parentheses, sqrt, round, abs.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The math expression to evaluate, e.g. '(15 * 23.50) * 1.085'",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "search_knowledge",
        "description": "Look up factual knowledge about a topic such as countries, science facts, or general knowledge.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The topic to look up, e.g. 'South Korea' or 'water boiling point'",
                }
            },
            "required": ["topic"],
        },
    },
    {
        "name": "unit_converter",
        "description": "Convert between units. Supports temperature (C/F/K), length (m/km/mi/ft), weight (kg/lb/g/oz), volume (L/gal/cup/ml).",
        "parameters": {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The numeric value to convert"},
                "from_unit": {"type": "string", "description": "Source unit (e.g. 'F', 'kg', 'cup')"},
                "to_unit": {"type": "string", "description": "Target unit (e.g. 'C', 'lb', 'ml')"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    },
    {
        "name": "date_info",
        "description": "Get date-related information such as days in a year, leap year checks, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The date question, e.g. 'days in 3 years' or 'is 2024 a leap year'",
                }
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Agent implementations — one per provider
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "Use tools when calculation, conversion, or factual lookup is needed. "
    "For simple factual questions you already know (like common knowledge), "
    "answer directly without tools. "
    "Always provide a clear final answer."
)


def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Execute a tool by name with given arguments."""
    fn = TOOLS_REGISTRY.get(name)
    if fn is None:
        return f"Error: Unknown tool '{name}'"
    try:
        return fn(**args)
    except Exception as exc:
        return f"Error executing {name}: {exc}"


def _run_anthropic(query: str, model: str) -> AgentOutput:
    """Run agent using Anthropic API."""
    import anthropic

    client = anthropic.Anthropic()
    messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
    steps: list[TrajectoryStep] = []
    total_input_tokens = 0
    total_output_tokens = 0
    step_idx = 0
    start_time = time.time()

    for _iteration in range(6):  # max 6 LLM calls
        iter_start = time.time()
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS_ANTHROPIC,
                messages=messages,
            )
        except Exception as exc:
            elapsed = (time.time() - start_time) * 1000
            return AgentOutput(
                output="",
                steps=steps,
                metadata=AgentMetadata(
                    total_tokens=total_input_tokens + total_output_tokens,
                    prompt_tokens=total_input_tokens,
                    completion_tokens=total_output_tokens,
                    cost=calculate_cost(model, total_input_tokens, total_output_tokens),
                    duration_ms=elapsed,
                ),
                success=False,
                error=str(exc),
            )
        iter_ms = (time.time() - iter_start) * 1000

        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Record LLM call step
        steps.append(
            TrajectoryStep(
                step_index=step_idx,
                step_type=StepType.LLM_CALL,
                name="llm_call",
                parameters={"model": model, "iteration": _iteration},
                output=response.stop_reason,
                duration_ms=iter_ms,
                tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
        )
        step_idx += 1

        # Check if we got a final text response (no tool use)
        if response.stop_reason == "end_turn":
            text_parts = [b.text for b in response.content if b.type == "text"]
            final_text = "\n".join(text_parts)
            elapsed = (time.time() - start_time) * 1000
            return AgentOutput(
                output=final_text,
                steps=steps,
                metadata=AgentMetadata(
                    total_tokens=total_input_tokens + total_output_tokens,
                    prompt_tokens=total_input_tokens,
                    completion_tokens=total_output_tokens,
                    cost=calculate_cost(model, total_input_tokens, total_output_tokens),
                    duration_ms=elapsed,
                ),
                success=True,
            )

        # Process tool use blocks
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_use_blocks:
            # No tool use and not end_turn — extract any text
            text_parts = [b.text for b in response.content if b.type == "text"]
            final_text = "\n".join(text_parts) if text_parts else ""
            elapsed = (time.time() - start_time) * 1000
            return AgentOutput(
                output=final_text,
                steps=steps,
                metadata=AgentMetadata(
                    total_tokens=total_input_tokens + total_output_tokens,
                    prompt_tokens=total_input_tokens,
                    completion_tokens=total_output_tokens,
                    cost=calculate_cost(model, total_input_tokens, total_output_tokens),
                    duration_ms=elapsed,
                ),
                success=True,
            )

        # Build assistant message with all content blocks
        messages.append({"role": "assistant", "content": response.content})

        # Execute each tool and build tool_result blocks
        tool_results = []
        for tool_block in tool_use_blocks:
            tool_name = tool_block.name
            tool_args = tool_block.input if isinstance(tool_block.input, dict) else {}
            tool_start = time.time()
            result = _execute_tool(tool_name, tool_args)
            tool_ms = (time.time() - tool_start) * 1000

            steps.append(
                TrajectoryStep(
                    step_index=step_idx,
                    step_type=StepType.TOOL_CALL,
                    name=tool_name,
                    parameters=tool_args,
                    output=result,
                    duration_ms=tool_ms,
                    tokens=0,
                )
            )
            step_idx += 1

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result,
                }
            )

        messages.append({"role": "user", "content": tool_results})

    # Max iterations reached
    elapsed = (time.time() - start_time) * 1000
    return AgentOutput(
        output="Max iterations reached without final answer.",
        steps=steps,
        metadata=AgentMetadata(
            total_tokens=total_input_tokens + total_output_tokens,
            prompt_tokens=total_input_tokens,
            completion_tokens=total_output_tokens,
            cost=calculate_cost(model, total_input_tokens, total_output_tokens),
            duration_ms=elapsed,
        ),
        success=False,
        error="Max iterations reached",
    )


def _run_google(query: str, model: str) -> AgentOutput:
    """Run agent using Google GenAI API."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY", os.environ.get("GEMINI_API_KEY", "")))

    # Build tool declarations for Google
    tool_declarations = []
    for schema in TOOL_SCHEMAS_GOOGLE:
        tool_declarations.append(
            types.FunctionDeclaration(
                name=schema["name"],
                description=schema["description"],
                parameters=schema["parameters"],
            )
        )
    google_tools = types.Tool(function_declarations=tool_declarations)

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=query)])
    ]
    steps: list[TrajectoryStep] = []
    total_input_tokens = 0
    total_output_tokens = 0
    step_idx = 0
    start_time = time.time()

    for _iteration in range(6):
        iter_start = time.time()
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=[google_tools],
                    max_output_tokens=1024,
                ),
            )
        except Exception as exc:
            elapsed = (time.time() - start_time) * 1000
            return AgentOutput(
                output="",
                steps=steps,
                metadata=AgentMetadata(
                    total_tokens=total_input_tokens + total_output_tokens,
                    prompt_tokens=total_input_tokens,
                    completion_tokens=total_output_tokens,
                    cost=calculate_cost(model, total_input_tokens, total_output_tokens),
                    duration_ms=elapsed,
                ),
                success=False,
                error=str(exc),
            )
        iter_ms = (time.time() - iter_start) * 1000

        # Token usage
        if response.usage_metadata:
            total_input_tokens += response.usage_metadata.prompt_token_count or 0
            total_output_tokens += response.usage_metadata.candidates_token_count or 0

        steps.append(
            TrajectoryStep(
                step_index=step_idx,
                step_type=StepType.LLM_CALL,
                name="llm_call",
                parameters={"model": model, "iteration": _iteration},
                output="function_call" if _has_function_calls(response) else "text",
                duration_ms=iter_ms,
                tokens=(response.usage_metadata.total_token_count or 0) if response.usage_metadata else 0,
            )
        )
        step_idx += 1

        # Check for function calls
        if not _has_function_calls(response):
            # Final text response
            final_text = response.text or ""
            elapsed = (time.time() - start_time) * 1000
            return AgentOutput(
                output=final_text,
                steps=steps,
                metadata=AgentMetadata(
                    total_tokens=total_input_tokens + total_output_tokens,
                    prompt_tokens=total_input_tokens,
                    completion_tokens=total_output_tokens,
                    cost=calculate_cost(model, total_input_tokens, total_output_tokens),
                    duration_ms=elapsed,
                ),
                success=True,
            )

        # Process function calls
        candidate = response.candidates[0]
        model_parts = list(candidate.content.parts)
        contents.append(types.Content(role="model", parts=model_parts))

        function_response_parts = []
        for part in model_parts:
            if part.function_call:
                fc = part.function_call
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}

                tool_start = time.time()
                result = _execute_tool(tool_name, tool_args)
                tool_ms = (time.time() - tool_start) * 1000

                steps.append(
                    TrajectoryStep(
                        step_index=step_idx,
                        step_type=StepType.TOOL_CALL,
                        name=tool_name,
                        parameters=tool_args,
                        output=result,
                        duration_ms=tool_ms,
                        tokens=0,
                    )
                )
                step_idx += 1

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=tool_name,
                        response={"result": result},
                    )
                )

        contents.append(types.Content(role="user", parts=function_response_parts))

    elapsed = (time.time() - start_time) * 1000
    return AgentOutput(
        output="Max iterations reached without final answer.",
        steps=steps,
        metadata=AgentMetadata(
            total_tokens=total_input_tokens + total_output_tokens,
            prompt_tokens=total_input_tokens,
            completion_tokens=total_output_tokens,
            cost=calculate_cost(model, total_input_tokens, total_output_tokens),
            duration_ms=elapsed,
        ),
        success=False,
        error="Max iterations reached",
    )


def _has_function_calls(response: Any) -> bool:
    """Check if a Google GenAI response contains function calls."""
    if not response.candidates:
        return False
    for part in response.candidates[0].content.parts:
        if part.function_call:
            return True
    return False


# ---------------------------------------------------------------------------
# Public factory + callable
# ---------------------------------------------------------------------------


def create_agent(
    provider: str | None = None,
    model: str | None = None,
) -> Any:
    """Create an agentrial-compatible agent callable.

    Args:
        provider: "anthropic" or "google". Defaults to AGENT_PROVIDER env var.
        model: Model name. Defaults to AGENT_MODEL env var.

    Returns:
        Callable[[AgentInput], AgentOutput]
    """
    provider = provider or os.environ.get("AGENT_PROVIDER", "anthropic")
    model = model or os.environ.get("AGENT_MODEL", "claude-3-haiku-20240307")

    def agent(input: AgentInput) -> AgentOutput:
        if provider == "google":
            return _run_google(input.query, model)  # type: ignore[arg-type]
        else:
            return _run_anthropic(input.query, model)  # type: ignore[arg-type]

    return agent


# Default agent callable — reads provider/model from env vars
agent = create_agent()
