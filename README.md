<p align="center">
  <h1 align="center">agentrial</h1>
  <p align="center">
    <strong>Statistical evaluation framework for AI agents</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/agentrial/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agentrial?color=blue"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
  </p>
</p>

Your agent passes Monday, fails Wednesday. **agentrial** tells you why.

AI agents are non-deterministic. A single test run tells you nothing. agentrial runs your agent N times, computes confidence intervals on pass rates, tracks real API costs, and pinpoints which step in the trajectory causes failures — so you ship agents that work reliably, not just once.

```
╭─────────────────────────────────────────────────────────────────────────────╮
│ my-agent-tests - PASSED                                                     │
╰───────────────────────────────────────────────────────-── Threshold: 85.0% ─╯
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Test Case              ┃ Pass Rate┃ 95% CI          ┃ Avg Cost ┃ Avg Latency┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━┩
│ easy-multiply          │   100.0% │ (72.2%-100.0%)  │  $0.0005 │      1.59s │
│ medium-inference       │   100.0% │ (72.2%-100.0%)  │  $0.0006 │      2.61s │
│ hard-multi-step        │   100.0% │ (72.2%-100.0%)  │  $0.0010 │      3.52s │
└────────────────────────┴──────────┴─────────────────┴──────────┴────────────┘

Overall Pass Rate: 100.0% (72.2%-100.0%)
Total Cost: $0.06
```

---

## Table of Contents

- [Why agentrial?](#why-agentrial)
- [Quick Start](#quick-start)
- [Writing Tests](#writing-tests)
- [Wrapping Your Agent](#wrapping-your-agent)
- [CLI Reference](#cli-reference)
- [Fluent Assertion API](#fluent-assertion-api)
- [CI/CD Integration](#cicd-integration)
- [Statistical Methods](#statistical-methods)
- [Architecture](#architecture)
- [Real-World Results](#real-world-results)
- [Supported Models](#supported-models)
- [Contributing](#contributing)
- [License](#license)

---

## Why agentrial?

Testing AI agents is fundamentally different from testing deterministic software. The same input can produce different tool calls, different reasoning paths, and different outputs. A single "it passed" run is meaningless.

agentrial solves this with:

| Feature | What it does |
|---|---|
| **Multi-trial execution** | Run every test N times automatically |
| **Wilson confidence intervals** | Statistically accurate pass rates, even with small sample sizes |
| **Step-level failure attribution** | Identifies *which tool call* diverges between passed and failed runs |
| **Real cost tracking** | Computes actual API costs from model metadata (supports 40+ models) |
| **Regression detection** | Fisher exact test catches reliability drops between versions |
| **CI/CD integration** | GitHub Action that blocks PRs when agent quality degrades |

---

## Quick Start

### Install

```bash
pip install agentrial
```

For LangGraph support:

```bash
pip install agentrial[langgraph]
```

### Create a test file

Create `agentrial.yml` in your project root:

```yaml
suite: my-agent-tests
agent: my_module.agent       # Python import path to your wrapped agent
trials: 10
threshold: 0.85              # Minimum pass rate to consider the suite passing

cases:
  - name: basic-math
    input:
      query: "What is 15 * 37?"
    expected:
      output_contains: ["555"]
      tool_calls:
        - tool: calculate
```

### Run

```bash
agentrial run
```

Or initialize a sample project:

```bash
agentrial init
```

---

## Writing Tests

Tests are defined in YAML files (`agentrial.yml`, or any `test_*.yml` / `test_*.yaml` file).

### Suite-level configuration

```yaml
suite: my-suite              # Suite name
agent: my_module.agent       # Python import path to wrapped agent callable
trials: 10                   # Number of trials per test case
threshold: 0.85              # Minimum overall pass rate (0.0 - 1.0)
```

### Test case options

Each test case supports a range of assertion types:

```yaml
cases:
  - name: test-name
    input:
      query: "User question"
      context:                   # Optional context dict passed to agent
        user_id: "123"

    expected:
      # String matching (AND logic — all must be present)
      output_contains: ["expected", "words"]

      # String matching (OR logic — at least one must be present)
      output_contains_any: ["option1", "option2"]

      # Regex pattern
      regex: "\\d+ results found"

      # Tool call assertions
      tool_calls:
        - tool: search
          params_contain:
            query: "expected search term"

    # Cost and latency constraints (per trial)
    max_cost: 0.05
    max_latency: 5000          # milliseconds

    # Step-level expectations
    step_expectations:
      - step_index: 0
        tool_name: search
        params_contain:
          query: "search term"
        output_contains: ["result"]
```

### Multiple test files

agentrial auto-discovers test files in the given directory:

```bash
agentrial run tests/          # Discovers test_*.yml, test_*.yaml, test_*.py
agentrial run agentrial.yml   # Run a specific file
```

---

## Wrapping Your Agent

agentrial needs a callable that takes `AgentInput` and returns `AgentOutput`. Use an adapter to wrap your framework's agent.

### LangGraph

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from agentrial.runner.adapters import wrap_langgraph_agent

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
graph = create_react_agent(llm, tools=[calculate])

# Export this — it's what agentrial.yml points to
agent = wrap_langgraph_agent(graph)
```

Then reference it in your YAML:

```yaml
agent: my_module.agent
```

The LangGraph adapter automatically captures:
- Full trajectory (every tool call and LLM response)
- Token usage per step
- Real API cost from model pricing data
- Execution duration

### Custom agents

Implement the agent protocol directly:

```python
from agentrial.types import AgentInput, AgentOutput, AgentMetadata

def agent(input: AgentInput) -> AgentOutput:
    # Your agent logic here
    return AgentOutput(
        output="result",
        steps=[],
        metadata=AgentMetadata(
            total_tokens=100,
            cost=0.001,
            duration_ms=500.0,
        ),
        success=True,
    )
```

---

## CLI Reference

```bash
# Run all tests in current directory
agentrial run

# Run specific file or directory
agentrial run tests/
agentrial run agentrial.yml

# Override trial count and threshold
agentrial run --trials 20 --threshold 0.9

# Export results to JSON
agentrial run -o results.json

# Output JSON to stdout
agentrial run --json

# Compare current results against a baseline
agentrial compare results.json --baseline baseline.json

# Save a baseline
agentrial baseline results.json -o baseline.json

# Show current configuration
agentrial config

# Initialize sample project
agentrial init
```

### Options

| Flag | Short | Description | Default |
|---|---|---|---|
| `--config` | `-c` | Path to config file | `agentrial.yml` |
| `--trials` | `-n` | Number of trials per test case | `10` |
| `--threshold` | `-t` | Minimum pass rate (0.0-1.0) | `0.85` |
| `--output` | `-o` | JSON output file path | — |
| `--json` | | Output JSON to stdout | `false` |

---

## Fluent Assertion API

For Python-defined tests, agentrial provides a chainable assertion builder:

```python
from agentrial import expect

result = agent(AgentInput(query="Book a flight to Rome"))

# Chain assertions fluently
expect(result) \
    .succeeded() \
    .output.contains("confirmed", "Rome") \
    .tool_called("search_flights") \
    .step(0).params_contain(destination="FCO") \
    .cost_below(0.15) \
    .latency_below(5000) \
    .tokens_below(3000) \
    .trajectory_length(min_steps=2, max_steps=10)
```

### Available assertions

| Method | Description |
|---|---|
| `.succeeded()` | Agent execution completed without error |
| `.output.contains(*strings)` | Output contains all substrings (AND) |
| `.output.equals(string)` | Output exactly matches string |
| `.output.matches(regex)` | Output matches regex pattern |
| `.output.length_between(min, max)` | Output length within bounds |
| `.tool_called(name, params_contain={})` | Specific tool was called with params |
| `.step(i).tool_name(name)` | Step at index is a call to named tool |
| `.step(i).params_contain(**kv)` | Step parameters contain expected values |
| `.step(i).output_contains(*strings)` | Step output contains substrings |
| `.cost_below(max_usd)` | Total cost under threshold |
| `.latency_below(max_ms)` | Total latency under threshold |
| `.tokens_below(max_tokens)` | Total tokens under threshold |
| `.trajectory_length(min, max)` | Number of steps within bounds |
| `.passed()` | Returns `True` if all assertions passed |
| `.get_failures()` | Returns list of failure messages |

---

## CI/CD Integration

### GitHub Actions (simple)

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
      - run: pip install agentrial[langgraph] langchain-anthropic
      - run: pip install -e .
      - run: agentrial run --trials 10 --threshold 0.85 -o results.json
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: agentrial-results
          path: results.json
```

### Regression detection in CI

Run against a saved baseline to catch reliability drops:

```yaml
      - run: agentrial run -o results.json
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      - run: agentrial compare results.json --baseline baseline.json
```

The `compare` command uses Fisher's exact test (p < 0.05) to detect statistically significant regressions. Exit code 1 if a regression is found.

---

## Statistical Methods

agentrial uses real statistical tests, not simple averages.

### Pass rate confidence intervals

**Wilson score interval** — accurate for small sample sizes and extreme proportions (0% or 100%), unlike normal approximation which fails at boundaries.

```
10 trials, 9 passes → 90.0% (59.6% - 98.2%)
10 trials, 10 passes → 100.0% (72.2% - 100.0%)
```

### Cost and latency confidence intervals

**Bootstrap resampling** (500 iterations) — non-parametric, no normality assumption required. Reports mean with 95% CI for cost and latency metrics.

### Regression detection

| Test | Use case |
|---|---|
| **Fisher exact test** | Compare pass rates between two runs (p < 0.05) |
| **Mann-Whitney U test** | Compare cost/latency distributions |
| **Benjamini-Hochberg correction** | Controls false discovery rate when comparing multiple metrics |

### Step-level failure attribution

When tests fail, agentrial analyzes trajectory divergence between passing and failing trials:
1. Groups trials by pass/fail
2. At each step, compares the distribution of tool calls
3. Uses Fisher exact test to identify the step with statistically significant divergence
4. Reports the divergent step with a human-readable recommendation

---

## Architecture

```
agentrial/
├── cli.py                  # Click CLI — run, compare, baseline, config, init
├── config.py               # YAML config loading and test file discovery
├── types.py                # Dataclasses: AgentInput, AgentOutput, TestCase, etc.
├── runner/
│   ├── engine.py           # MultiTrialEngine — orchestrates N trials per test
│   ├── trajectory.py       # TrajectoryRecorder — captures steps, tokens, cost
│   ├── otel.py             # OpenTelemetry span capture for any framework
│   └── adapters/
│       ├── base.py         # BaseAdapter protocol
│       ├── langgraph.py    # LangGraph adapter with callback-based capture
│       └── pricing.py      # Model pricing for 40+ LLMs
├── evaluators/
│   ├── exact.py            # contains, regex, tool_called, exact_match
│   ├── expect.py           # Fluent assertion API
│   ├── functional.py       # Custom check functions, range checks
│   └── step_eval.py        # Per-step and trajectory evaluation
├── metrics/
│   ├── basic.py            # Pass rate, cost, latency, token efficiency
│   ├── statistical.py      # Wilson CI, bootstrap, Fisher, Mann-Whitney, BH
│   └── trajectory.py       # Failure attribution via divergence analysis
└── reporters/
    ├── terminal.py         # Rich terminal output with colored tables
    └── json_report.py      # JSON export, load, and comparison
```

---

## Real-World Results

Tested with **Claude 3 Haiku** on a 3-tool agent (calculator, country lookup, temperature conversion) — 100 trials:

| Test Complexity | Pass Rate | 95% CI | Avg Cost | Avg Latency | Avg Tokens |
|---|---|---|---|---|---|
| Easy (direct tool call) | 100% | 72.2% - 100% | $0.0005 | 1.6s | 1,513 |
| Medium (inference + tool) | 100% | 72.2% - 100% | $0.0006 | 2.6s | 1,926 |
| Hard (multi-step reasoning) | 100% | 72.2% - 100% | $0.0010 | 3.5s | 2,986 |

**100 trials total. $0.06 total cost. Full trajectory capture.**

See the complete example in [`examples/langgraph_haiku/`](examples/langgraph_haiku/).

---

## Supported Models

agentrial has built-in pricing data for cost tracking across major providers:

| Provider | Models |
|---|---|
| **Anthropic** | Claude 3 Haiku, Sonnet, Opus (all versions), Claude 3.5, Claude 4 |
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-4, GPT-3.5 Turbo |
| **Google** | Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.0 Pro |
| **Mistral** | Large, Medium, Small |

Cost is extracted automatically from model response metadata. No configuration needed.

---

## Supported Frameworks

| Framework | Status | Notes |
|---|---|---|
| **LangGraph** | Native adapter | Full trajectory capture, callbacks, token tracking |
| **Any OpenTelemetry-instrumented agent** | Supported | Automatic span capture via OTel SDK |
| **Custom** | Supported | Implement `AgentInput -> AgentOutput` protocol |

---

## Contributing

```bash
# Clone and install
git clone https://github.com/alepot55/agentrial.git
cd agentrial
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check .

# Type check
mypy agentrial/
```

---

## License

[MIT](LICENSE)
