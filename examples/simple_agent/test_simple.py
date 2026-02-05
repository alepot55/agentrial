"""Example Python test suite for simple_qa_agent."""

from agenteval import Suite, TestCase, expect
from agenteval.types import AgentInput

# Create a test suite
suite = Suite(
    name="simple-qa-python-tests",
    agent="examples.simple_agent.simple_qa_agent",
    trials=5,
    threshold=0.8,
)


@suite.case
def test_greeting(result):
    """Test that the agent responds to greetings."""
    expect(result).output.contains("Hello", "help")
    expect(result).succeeded()
    expect(result).cost_below(0.01)


@suite.case
def test_weather_query(result):
    """Test weather-related queries."""
    expect(result).output.contains("weather")
    expect(result).tool_called("search")
    expect(result).latency_below(500)


# Add test cases programmatically
suite.cases.append(
    TestCase(
        name="flight-search-programmatic",
        input=AgentInput(query="Find flights from Rome to Tokyo"),
        expected=None,  # We'll use the validator below
    )
)


# Custom validation function
def validate_flight_search(output):
    """Validate flight search response."""
    e = expect(output)
    e.output.contains("flight")
    e.tool_called("search")
    e.cost_below(0.05)
    return e.passed()
