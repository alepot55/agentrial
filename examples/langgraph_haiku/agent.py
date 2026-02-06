"""LangGraph agent with three tools for Agentrial testing.

This agent uses Claude 3 Haiku via langchain-anthropic and has tools for:
- Mathematical calculations
- Country information lookup
- Temperature conversion

Requirements:
    pip install agentrial[langgraph] langchain-anthropic

Usage:
    export ANTHROPIC_API_KEY="your-key"
    agentrial run
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agentrial.runner.adapters import wrap_langgraph_agent


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate (e.g., "15 * 37", "2 ** 10")

    Returns:
        The result of the calculation as a string.
    """
    try:
        allowed_names = {
            "__builtins__": {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        }
        result = eval(expression, allowed_names, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def lookup_country_info(country: str) -> str:
    """Look up capital and population for a country.

    Args:
        country: The name of the country to look up.

    Returns:
        Information about the country's capital and population.
    """
    data = {
        "Japan": {"capital": "Tokyo", "population": "125M"},
        "France": {"capital": "Paris", "population": "68M"},
        "Brazil": {"capital": "Brasilia", "population": "214M"},
        "Italy": {"capital": "Rome", "population": "59M"},
        "Germany": {"capital": "Berlin", "population": "84M"},
        "India": {"capital": "New Delhi", "population": "1.4B"},
        "Australia": {"capital": "Canberra", "population": "26M"},
        "Canada": {"capital": "Ottawa", "population": "40M"},
        "Mexico": {"capital": "Mexico City", "population": "128M"},
        "South Korea": {"capital": "Seoul", "population": "52M"},
    }
    info = data.get(country)
    if info:
        return f"{country}: capital is {info['capital']}, population is {info['population']}"
    return f"No data found for {country}"


@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius (C), Fahrenheit (F), and Kelvin (K).

    Args:
        value: The temperature value to convert.
        from_unit: Source unit - 'C' for Celsius, 'F' for Fahrenheit, 'K' for Kelvin.
        to_unit: Target unit - 'C' for Celsius, 'F' for Fahrenheit, 'K' for Kelvin.

    Returns:
        The converted temperature value with units.
    """
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()

    # Convert to Celsius first
    if from_unit == "C":
        celsius = value
    elif from_unit == "F":
        celsius = (value - 32) * 5 / 9
    elif from_unit == "K":
        celsius = value - 273.15
    else:
        return f"Error: Unknown source unit '{from_unit}'. Use C, F, or K."

    # Convert from Celsius to target
    if to_unit == "C":
        result = celsius
    elif to_unit == "F":
        result = celsius * 9 / 5 + 32
    elif to_unit == "K":
        result = celsius + 273.15
    else:
        return f"Error: Unknown target unit '{to_unit}'. Use C, F, or K."

    return f"{value}{from_unit} = {result:.2f}{to_unit}"


def create_agent():
    """Create and return the LangGraph agent."""
    llm = ChatAnthropic(
        model="claude-3-haiku-20240307",
        temperature=0,
    )
    tools = [calculate, lookup_country_info, convert_temperature]
    graph = create_react_agent(llm, tools)
    return graph


# Create the graph and wrap for agentrial
graph = create_agent()
agent = wrap_langgraph_agent(graph)


if __name__ == "__main__":
    from agentrial.types import AgentInput

    test_input = AgentInput(query="What is 15 * 37?")
    result = agent(test_input)

    print(f"Output: {result.output}")
    print(f"Steps: {len(result.steps)}")
    print(f"Tokens: {result.metadata.total_tokens}")
    print(f"Cost: ${result.metadata.cost:.6f}")
