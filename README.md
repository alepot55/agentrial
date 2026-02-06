<p align="center">
  <h1 align="center">agentrial</h1>
  <p align="center">
    <strong>The pytest for AI agents. Statistical evaluation with confidence intervals and failure attribution.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/agentrial/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agentrial?color=blue"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
  </p>
</p>

Your agent passes Monday, fails Wednesday. Same prompt, same model. **agentrial** tells you why.

---

## Quickstart

```bash
pip install agentrial
agentrial init
agentrial run
```

That's it. You'll see real results in seconds:

```
╭──────────────────────────────────────────────────────────────────────╮
│ sample-demo - PASSED                                                 │
╰───────────────────────────────────────────────────── Threshold: 80% ─╯
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Test Case        ┃ Pass Rate ┃ 95% CI           ┃ Avg Cost ┃ Avg Latency ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ greeting         │    100.0% │ (56.6%-100.0%)   │  $0.0000 │         0ms │
│ capital-france   │    100.0% │ (56.6%-100.0%)   │  $0.0000 │         0ms │
│ capital-japan    │    100.0% │ (56.6%-100.0%)   │  $0.0000 │         0ms │
│ basic-math       │    100.0% │ (56.6%-100.0%)   │  $0.0000 │         0ms │
└──────────────────┴───────────┴──────────────────┴──────────┴─────────────┘

Overall Pass Rate: 100.0% (85.0%-100.0%)
Total Cost: $0.0000
```

Replace `sample_agent.py` with your own agent, update `tests/test_sample.yml`, and you're evaluating real agents.

---

## What it does

- **Multi-trial execution** — Run every test N times automatically. A single pass means nothing for non-deterministic agents.
- **Wilson confidence intervals** — Statistically accurate pass rates, even with small samples and extreme proportions (0% or 100%).
- **Step-level failure attribution** — Pinpoints *which tool call* diverges between passing and failing runs using Fisher exact test.
- **Real cost tracking** — Actual API costs from model metadata, 40+ models supported across Anthropic, OpenAI, Google, Mistral.
- **Regression detection** — Fisher exact test catches reliability drops between versions. Blocks PRs in CI when quality degrades.
- **Local-first** — Your data never leaves your machine. No accounts, no SaaS, no telemetry.

---

## Why this exists

Every agent framework ships with benchmarks showing 90%+ accuracy. But run those same agents 100 times on the same task, and you'll see pass rates drop to 60-80% with wide variance. The benchmarks measure one run; production sees thousands.

No existing tool gives you statistically rigorous, framework-agnostic agent testing that runs in CI/CD. LangSmith requires a paid account and locks you to LangChain. Promptfoo doesn't do multi-trial with confidence intervals. DeepEval and Arize don't do trajectory-level failure attribution. agentrial fills that gap: open-source, free, local-first, works with any agent framework.

---

## How it compares

| Feature                      | agentrial | Promptfoo | LangSmith | DeepEval | Arize |
|------------------------------|-----------|-----------|-----------|----------|-------|
| Multi-trial with CI          | **Free**  | No        | $39/mo    | No       | No    |
| Confidence intervals         | Yes       | No        | No        | No       | No    |
| Trajectory step analysis     | Yes       | No        | Partial   | No       | Yes   |
| Failure attribution          | Yes       | No        | No        | No       | No    |
| Framework-agnostic (OTel)    | Yes       | Yes       | No        | Yes      | Yes   |
| Free CI/CD integration       | Yes       | Yes       | No        | No       | No    |
| Local-first (no data leaves) | Yes       | Yes       | No        | No       | No    |
| Cost-per-correct-answer      | Yes       | No        | No        | No       | No    |

---

## Writing Tests

Tests are YAML files. Define what your agent receives and what it should produce:

```yaml
suite: my-agent-tests
agent: my_module.agent       # Python import path to your wrapped agent
trials: 10
threshold: 0.85              # Minimum pass rate

cases:
  - name: basic-math
    input:
      query: "What is 15 * 37?"
    expected:
      output_contains: ["555"]
      tool_calls:
        - tool: calculate

  - name: capital-lookup
    input:
      query: "What is the capital of Japan?"
    expected:
      output_contains: ["Tokyo"]

  - name: error-handling
    input:
      query: "Divide 10 by 0"
    expected:
      output_contains_any: ["undefined", "cannot", "error"]
    max_cost: 0.05
    max_latency_ms: 5000
```

### All assertion types

```yaml
expected:
  output_contains: ["word1", "word2"]        # AND — all must be present
  output_contains_any: ["option1", "option2"] # OR — at least one
  exact_match: "exact output string"
  regex: "\\d+ results found"
  tool_calls:
    - tool: search
      params_contain:
        query: "expected term"

# Per-step expectations
step_expectations:
  - step_index: 0
    tool_name: search
    params_contain:
      query: "search term"
    output_contains: ["result"]
```

### Test discovery

agentrial auto-discovers test files:

```bash
agentrial run tests/          # Finds test_*.yml, test_*.yaml, test_*.py
agentrial run agentrial.yml   # Run a specific file
```

---

## Wrapping Your Agent

agentrial needs a callable: `AgentInput -> AgentOutput`. Use an adapter for your framework.

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

# This is what your YAML points to
agent = wrap_langgraph_agent(graph)
```

The LangGraph adapter automatically captures full trajectory, token usage, real API cost, and execution duration.

### Custom agents

Implement the protocol directly:

```python
from agentrial.types import AgentInput, AgentOutput, AgentMetadata

def agent(input: AgentInput) -> AgentOutput:
    # Your agent logic
    return AgentOutput(
        output="result",
        steps=[],
        metadata=AgentMetadata(total_tokens=100, cost=0.001, duration_ms=500.0),
        success=True,
    )
```

---

## Fluent Assertion API

For Python-defined tests:

```python
from agentrial import expect

result = agent(AgentInput(query="Book a flight to Rome"))

expect(result) \
    .succeeded() \
    .output.contains("confirmed", "Rome") \
    .tool_called("search_flights") \
    .step(0).params_contain(destination="FCO") \
    .cost_below(0.15) \
    .latency_below(5000)
```

| Method | Description |
|---|---|
| `.succeeded()` | Agent completed without error |
| `.output.contains(*strings)` | Output contains all substrings |
| `.output.equals(string)` | Exact match |
| `.output.matches(regex)` | Regex match |
| `.tool_called(name, params={})` | Tool was called with params |
| `.step(i).tool_name(name)` | Step i called named tool |
| `.cost_below(max_usd)` | Cost under threshold |
| `.latency_below(max_ms)` | Latency under threshold |
| `.tokens_below(max_tokens)` | Tokens under threshold |
| `.trajectory_length(min, max)` | Step count within bounds |
| `.passed()` | Returns `True` if all pass |
| `.get_failures()` | Returns failure messages |

---

## CLI Reference

```bash
agentrial init                          # Create sample project (ready to run)
agentrial run                           # Run all tests in current directory
agentrial run tests/                    # Run tests in specific directory
agentrial run --trials 20 --threshold 0.9  # Override settings
agentrial run -o results.json           # Export JSON report
agentrial run --json                    # JSON to stdout
agentrial compare results.json -b baseline.json  # Regression detection
agentrial baseline results.json         # Save baseline
agentrial config                        # Show configuration
```

| Flag | Short | Description | Default |
|---|---|---|---|
| `--config` | `-c` | Config file path | `agentrial.yml` |
| `--trials` | `-n` | Trials per test case | `10` |
| `--threshold` | `-t` | Min pass rate (0-1) | `0.85` |
| `--output` | `-o` | JSON output path | — |
| `--json` | | JSON to stdout | `false` |

---

## CI/CD Integration

### GitHub Actions

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
      - run: pip install agentrial && pip install -e .
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

```yaml
      - run: agentrial run -o results.json
      - run: agentrial compare results.json --baseline baseline.json
```

Fisher's exact test (p < 0.05) detects statistically significant regressions. Exit code 1 blocks the PR.

---

## Statistical Methods

agentrial uses real statistical tests, not simple averages.

| Method | What it does |
|---|---|
| **Wilson score interval** | Confidence intervals for pass rates — accurate at boundaries (0%, 100%) and small samples |
| **Bootstrap resampling** | CI for cost/latency — non-parametric, no normality assumption (500 iterations) |
| **Fisher exact test** | Regression detection — compares pass rates between two runs (p < 0.05) |
| **Mann-Whitney U test** | Compares cost/latency distributions between versions |
| **Benjamini-Hochberg** | Controls false discovery rate when comparing multiple metrics |

### Failure attribution

When tests fail, agentrial analyzes trajectory divergence:
1. Groups trials by pass/fail
2. At each step, compares distribution of tool calls
3. Fisher exact test identifies the step with significant divergence
4. Reports the divergent step with a recommendation

---

## Architecture

```
agentrial/
├── cli.py                  # Click CLI — run, compare, baseline, config, init
├── config.py               # YAML config loading and test file discovery
├── types.py                # AgentInput, AgentOutput, TestCase, Suite, etc.
├── runner/
│   ├── engine.py           # MultiTrialEngine — orchestrates N trials per test
│   ├── trajectory.py       # TrajectoryRecorder — captures steps, tokens, cost
│   ├── otel.py             # OpenTelemetry span capture for any framework
│   └── adapters/
│       ├── base.py         # BaseAdapter protocol
│       ├── langgraph.py    # LangGraph adapter (callbacks + trajectory)
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
    ├── terminal.py         # Rich terminal output
    └── json_report.py      # JSON export, load, comparison
```

---

## Supported Frameworks

| Framework | Status | Notes |
|---|---|---|
| **LangGraph** | Native adapter | Full trajectory, callbacks, token tracking |
| **Any OTel-instrumented agent** | Supported | Automatic span capture via OTel SDK |
| **Custom** | Supported | `AgentInput -> AgentOutput` protocol |

## Supported Models (cost tracking)

| Provider | Models |
|---|---|
| **Anthropic** | Claude 3 Haiku/Sonnet/Opus, Claude 3.5, Claude 4 |
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-3.5 Turbo |
| **Google** | Gemini 1.5 Pro/Flash, Gemini 1.0 Pro |
| **Mistral** | Large, Medium, Small |

---

## Contributing

```bash
git clone https://github.com/alepot55/agentrial.git
cd agentrial
pip install -e ".[dev]"
pytest
ruff check .
mypy agentrial/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

[MIT](LICENSE)
