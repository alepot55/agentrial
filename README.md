# AgentEval

[![CI](https://github.com/agenteval/agenteval/workflows/CI/badge.svg)](https://github.com/agenteval/agenteval/actions)
[![PyPI](https://img.shields.io/pypi/v/agenteval.svg)](https://pypi.org/project/agenteval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Your agent passes Monday, fails Wednesday. Same prompt. agenteval tells you why.**

```bash
pip install agenteval
```

## Quickstart

### 1. Define your agent

```python
# my_agent.py
from agenteval.types import AgentInput, AgentOutput, AgentMetadata, TrajectoryStep, StepType

def search_agent(input: AgentInput) -> AgentOutput:
    # Your agent logic - call LLM, use tools, etc.
    response = llm.chat(input.query)

    return AgentOutput(
        output=response.text,
        steps=[
            TrajectoryStep(
                step_index=0,
                step_type=StepType.TOOL_CALL,
                name="search_flights",
                parameters={"destination": "TYO"},
                output={"flights": [...]},
            ),
        ],
        metadata=AgentMetadata(
            total_tokens=response.usage.total_tokens,
            cost=response.usage.total_tokens * 0.00003,
        ),
    )
```

### 2. Write test cases

```yaml
# tests/test_search.yml
suite: search-agent
agent: my_agent.search_agent
trials: 10
threshold: 0.85

cases:
  - name: flight-search
    input:
      query: "Find cheapest flight from Rome to Tokyo in March"
    expected:
      output_contains: ["flight", "Tokyo"]
      tool_calls:
        - tool: search_flights
          params_contain: {destination: "TYO"}

  - name: hotel-search
    input:
      query: "Find hotels near Shibuya station"
    expected:
      output_contains: ["hotel", "Shibuya"]
```

### 3. Run evaluation

```bash
agenteval run tests/test_search.yml --trials 10
```

Output:

```
AgentEval v0.1.0

Running suite: search-agent (2 test cases, 10 trials each)

┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Test Case      ┃ Pass Rate ┃ 95% CI       ┃ Avg Cost ┃ Avg Latency ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ flight-search  │ 80%       │ (49%-94%)    │ $0.0023  │ 342ms       │
│ hotel-search   │ 70%       │ (40%-89%)    │ $0.0019  │ 287ms       │
└────────────────┴───────────┴──────────────┴──────────┴─────────────┘

Overall: 75% (58%-87%) | Threshold: 85% | Status: FAILED

Failure Attribution:
  flight-search: Step 1 (tool_selection) shows significant divergence (p=0.02)
    - Successful runs: search_flights (8/8)
    - Failed runs: search_hotels (2/2)

  hotel-search: Step 2 (parameter_extraction) shows significant divergence (p=0.04)
    - Successful runs: location="Shibuya" (7/7)
    - Failed runs: location="Tokyo" (3/3)

Total cost: $0.042 | Duration: 12.4s
```

The output shows:
- **Pass rate with confidence interval**: 80% (49%-94%) means the true pass rate is likely between 49% and 94%
- **Failure attribution**: Step 1 diverges between successful and failed runs - successful runs call `search_flights`, failed runs call `search_hotels`
- **Cost tracking**: Total spend across all trials

## CI/CD

Add to your GitHub workflow (`.github/workflows/agenteval.yml`):

```yaml
- uses: agenteval/agenteval@v1
  with:
    trials: 10
    threshold: 0.9
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

The action posts results as a PR comment and fails if pass rate drops below threshold.

## Why AgentEval?

- **Statistical rigor**: Wilson score confidence intervals, not single-run pass/fail. Know if your 80% pass rate is real or noise.

- **Trajectory analysis**: When tests fail, identifies which step diverged between successful and failed runs using Fisher's exact test.

- **Framework-agnostic**: Works with LangGraph, raw OpenAI calls, or any Python function. Uses OpenTelemetry spans as the universal integration point.

## LangGraph Integration

```python
from langgraph.graph import StateGraph
from agenteval.adapters.langgraph import wrap_langgraph_agent

graph = StateGraph(...)
compiled = graph.compile()
agent = wrap_langgraph_agent(compiled)  # Captures trajectory via OTel spans
```

## CLI Reference

```bash
agenteval run [PATH]           # Run tests (default: tests/)
agenteval run --trials 20      # Override trial count
agenteval run --threshold 0.9  # Override pass threshold
agenteval run -o results.json  # Export JSON for CI

agenteval compare --baseline baseline.json  # Compare against baseline
agenteval baseline results.json             # Save new baseline
agenteval init                              # Initialize project
```

## License

MIT
