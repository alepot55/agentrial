# agentrial

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

> Your agent passes Monday, fails Wednesday. agentrial tells you why.

Statistical evaluation framework for AI agents. Like pytest, but for non-deterministic systems.

## Why agentrial?

- **Statistical rigor**: Every test runs N trials with Wilson confidence intervals — not "it passed once"
- **Trajectory analysis**: Step-by-step failure attribution identifies exactly which tool call diverged
- **Cost tracking**: Real API costs from model metadata, not estimates
- **CI/CD ready**: GitHub Action that blocks PRs when reliability drops

## Quick Start

```bash
pip install agentrial
```

Create a test file `agentrial.yml`:

```yaml
suite: my-agent-tests
agent: my_agent.agent  # Python import path
trials: 10
threshold: 0.85

cases:
  - name: basic-math
    input:
      query: "What is 15 * 37?"
    expected:
      output_contains: ["555"]
      tool_calls:
        - tool: calculate
```

Run:

```bash
agentrial run
```

Output:

```
╭──────────────────────────────────────────────────────────────────────────────╮
│ my-agent-tests - PASSED                                                      │
╰────────────────────────────────────────────────────── Threshold: 85.0% ──────╯
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Test Case              ┃ Pass Rate┃ 95% CI          ┃ Avg Cost ┃ Avg Latency┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ basic-math             │   100.0% │ (72.2%-100.0%)  │  $0.0005 │      1.59s │
└────────────────────────┴──────────┴─────────────────┴──────────┴────────────┘

Overall Pass Rate: 100.0% (72.2%-100.0%)
Total Cost: $0.005
```

## Real-World Results

Tested with Claude 3 Haiku on a 3-tool agent (calculator, country lookup, temperature conversion):

| Test Complexity | Pass Rate | 95% CI | Avg Cost | Avg Latency | Avg Tokens |
|-----------------|-----------|--------|----------|-------------|------------|
| Easy (direct tool call) | 100% | 72.2%-100% | $0.0005 | 1.6s | 1,513 |
| Medium (inference + tool) | 100% | 72.2%-100% | $0.0006 | 2.6s | 1,926 |
| Hard (multi-step reasoning) | 100% | 72.2%-100% | $0.0010 | 3.5s | 2,986 |

**100 trials total, $0.06 total cost, full trajectory capture.**

## Test Case Options

```yaml
cases:
  - name: my-test
    input:
      query: "User question"
      context: {}  # Optional context dict
    expected:
      # All strings must be present (AND logic)
      output_contains: ["expected", "words"]

      # At least one string must be present (OR logic)
      output_contains_any: ["option1", "option2", "option3"]

      # Regex pattern
      regex: "\\d+ results found"

      # Expected tool calls
      tool_calls:
        - tool: search
          params_contain: {query: "expected"}
```

## CI/CD Integration

Add to `.github/workflows/agent-eval.yml`:

```yaml
name: Agent Evaluation
on: [push, pull_request]

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install agentrial
      - run: agentrial run --threshold 0.85
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## LangGraph Integration

```python
from langchain_anthropic import ChatAnthropic
from langgraph.prebuilt import create_react_agent
from agentrial.runner.adapters import wrap_langgraph_agent

# Your LangGraph agent
llm = ChatAnthropic(model="claude-3-haiku-20240307")
graph = create_react_agent(llm, tools=[...])

# Wrap for agentrial
agent = wrap_langgraph_agent(graph)
```

Then reference in your test file:

```yaml
agent: my_module.agent
```

## CLI Reference

```bash
agentrial run [PATH]           # Run tests (default: current directory)
agentrial run --trials 20      # Override trial count
agentrial run --threshold 0.9  # Override pass threshold
agentrial run -o results.json  # Export JSON report

agentrial compare -b baseline.json current.json  # Compare runs
agentrial init                                   # Initialize project
```

## Supported Frameworks

- LangGraph (native adapter with full trajectory capture)
- CrewAI (coming soon)
- AutoGen (coming soon)
- Pydantic AI (coming soon)

## Statistical Methods

- **Pass rate CI**: Wilson score interval (accurate for small N and extreme proportions)
- **Cost/latency CI**: Bootstrap resampling (500 iterations)
- **Regression detection**: Fisher exact test for pass rate, Mann-Whitney U for metrics
- **Failure attribution**: Trajectory divergence analysis between passed/failed trials

## License

MIT
