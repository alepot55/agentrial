# AgentEval

**Statistical evaluation framework for AI agents** - the pytest for agent trajectories.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Your agent passes Monday, fails Wednesday. Same prompt. Same model. Same code.

AI agents are inherently non-deterministic. A single test run tells you nothing about reliability. You need:

- **Multiple trials** to understand true performance
- **Confidence intervals** to know if changes are real improvements
- **Step-level analysis** to find where failures happen
- **CI/CD integration** to catch regressions before production

## The Solution

AgentEval runs your agent N times (default: 10), computes pass rates with 95% confidence intervals, and identifies which trajectory step most likely causes failures.

```bash
pip install agenteval
agenteval run tests/test_my_agent.yml
```

Output:
```
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Test Case         ┃ Pass Rate ┃ 95% CI       ┃ Avg Cost ┃ Avg Latency ┃ Avg Tokens ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ flight-search     │ 80.0%     │ (55%-93%)    │ $0.0012  │ 234ms       │ 156        │
│ weather-query     │ 100.0%    │ (74%-100%)   │ $0.0008  │ 189ms       │ 98         │
│ booking-flow      │ 60.0%     │ (31%-83%)    │ $0.0045  │ 892ms       │ 445        │
└───────────────────┴───────────┴──────────────┴──────────┴─────────────┴────────────┘

Overall Pass Rate: 80.0% (67%-89%)
Total Cost: $0.065
```

## Quick Start

### 1. Install

```bash
pip install agenteval
```

### 2. Define your agent

Your agent must accept `AgentInput` and return `AgentOutput`:

```python
# my_agent.py
from agenteval.types import AgentInput, AgentOutput, AgentMetadata

def my_agent(input: AgentInput) -> AgentOutput:
    # Your agent logic here
    result = call_llm(input.query)

    return AgentOutput(
        output=result.text,
        steps=result.trajectory,  # Optional: trajectory steps
        metadata=AgentMetadata(
            total_tokens=result.tokens,
            cost=result.cost,
            duration_ms=result.duration,
        ),
    )
```

### 3. Write test cases

**YAML format:**

```yaml
# tests/test_my_agent.yml
suite: my-agent-tests
agent: my_agent.my_agent
trials: 10
threshold: 0.85

cases:
  - name: basic-query
    input:
      query: "What is the capital of France?"
    expected:
      output_contains:
        - "Paris"

  - name: tool-usage
    input:
      query: "Search for flights to Tokyo"
    expected:
      tool_calls:
        - tool: search_flights
          params_contain:
            destination: "TYO"
```

**Python format:**

```python
# tests/test_my_agent.py
from agenteval import Suite, expect

suite = Suite(
    name="my-agent-tests",
    agent="my_agent.my_agent",
    trials=10,
    threshold=0.85,
)

@suite.case
def test_basic_query(result):
    expect(result).output.contains("Paris")
    expect(result).cost_below(0.01)
    expect(result).latency_below(1000)
```

### 4. Run

```bash
agenteval run tests/
```

## Framework Integration

### LangGraph

```python
from langgraph.graph import StateGraph
from agenteval.adapters.langgraph import wrap_langgraph_agent

# Your LangGraph
graph = StateGraph(...)
compiled = graph.compile()

# Wrap for AgentEval
agent = wrap_langgraph_agent(compiled)
```

## CLI Commands

```bash
# Run all tests
agenteval run

# Run specific test file
agenteval run tests/test_flight.yml

# Override trials and threshold
agenteval run --trials 20 --threshold 0.9

# Save results as JSON
agenteval run -o results.json

# Compare against baseline
agenteval compare results.json --baseline baseline.json

# Save new baseline
agenteval baseline results.json

# Initialize project
agenteval init
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/agenteval.yml
name: AgentEval

on: [pull_request]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - run: pip install agenteval
      - run: agenteval run --threshold 0.85
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Statistical Methods

### Pass Rate Confidence Intervals

AgentEval uses the **Wilson score interval**, which is accurate for small samples and extreme proportions (unlike naive normal approximation).

With N=10 trials:
- 70% observed → 95% CI: 40%-89%
- 100% observed → 95% CI: 74%-100%

### Regression Detection

Compares runs using **Fisher's exact test** (for pass rates) and **Mann-Whitney U** (for latency/cost). Detects significant regressions at p<0.05.

### Failure Attribution

When tests fail, AgentEval analyzes which trajectory step shows significant divergence between successful and failed runs, helping you pinpoint the root cause.

## Configuration

Create `agenteval.yml` in your project root:

```yaml
trials: 10
threshold: 0.85
output_format: terminal  # or "json"
verbose: false
```

## Why AgentEval?

| Feature | AgentEval | LangSmith | Promptfoo | DeepEval |
|---------|-----------|-----------|-----------|----------|
| Multi-trial by default | Yes | No | No | No |
| Confidence intervals | Yes | No | No | No |
| Step-level analysis | Yes | Partial | No | No |
| Framework agnostic | Yes | LangChain | Yes | Yes |
| Local-first | Yes | Cloud | Yes | Yes |
| Statistical regression detection | Yes | No | No | No |

## License

MIT

## Contributing

Contributions welcome! Please read our contributing guide and submit PRs.
