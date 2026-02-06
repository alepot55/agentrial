<p align="center">
  <h1 align="center">agentrial</h1>
  <p align="center">
    <strong>The pytest for AI agents. Run your agent 100 times, get confidence intervals instead of anecdotes.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/agentrial/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agentrial?color=blue"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
    <a href="https://github.com/alepot55/agentrial/actions"><img alt="Tests" src="https://img.shields.io/badge/tests-349%20passed-brightgreen"></a>
  </p>
</p>

Your agent passes Monday, fails Wednesday. Same prompt, same model. LLMs show up to [72% variance across runs](https://arxiv.org/abs/2407.02100) even at temperature=0.

**agentrial** runs your agent N times and gives you statistics, not luck.

```bash
pip install agentrial
agentrial init
agentrial run
```

```
╭──────────────────────────────────────────────────────────────────────╮
│ my-agent - FAILED                                                    │
╰───────────────────────────────────────────────────── Threshold: 85% ─╯
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Test Case            ┃ Pass Rate ┃ 95% CI           ┃ Avg Cost ┃ Avg Latency ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ easy-multiply        │    100.0% │ (72.2%-100.0%)   │  $0.0005 │       320ms │
│ tool-selection       │     90.0% │ (59.6%-98.2%)    │  $0.0006 │       450ms │
│ multi-step-task      │     70.0% │ (39.7%-89.2%)    │  $0.0011 │       890ms │
│ ambiguous-query      │     50.0% │ (23.7%-76.3%)    │  $0.0008 │       670ms │
└──────────────────────┴───────────┴──────────────────┴──────────┴─────────────┘

Failure Attribution:
  tool-selection: Step 0 — called 'calculate' instead of 'lookup_country_info' (p=0.003)
  multi-step-task: Step 2 — missing second tool call 'calculate' after lookup (p=0.01)
  ambiguous-query: Step 0 — tool selection inconsistent across runs (p<0.001)

Overall Pass Rate: 77.5% (68.4%-84.5%) — BELOW THRESHOLD
Total Cost: $0.0600
```

That 100% on `easy-multiply`? Wilson CI says it's actually 72-100% with 10 trials. That `multi-step-task` at 70%? Step 2 is the bottleneck. Now you know what to fix.

---

## Why this exists

Every agent framework demos 90%+ accuracy. Run those agents 100 times on the same task, pass rates drop to 60-80% with wide variance. Benchmarks measure one run; production sees thousands.

No existing tool combines trajectory evaluation, multi-trial statistics, and CI/CD integration in a single open-source package. LangSmith requires paid accounts and LangChain lock-in. Promptfoo doesn't do multi-trial with confidence intervals. DeepEval and Arize don't do step-level failure attribution.

agentrial fills that gap: open-source, free, local-first, works with any framework.

---

## Core features

**Statistical rigor by default.** Every evaluation runs N trials with Wilson confidence intervals. Bootstrap resampling for cost/latency. Benjamini-Hochberg correction for multiple comparisons. No single-run pass/fail.

**Step-level failure attribution.** When tests fail, agentrial compares trajectories from passing and failing runs. Fisher exact test identifies the specific step where behavior diverges. You see "Step 2 tool selection is the problem" instead of "test failed."

**Real cost tracking.** Token usage from API response metadata, not estimates. 40+ models across Anthropic, OpenAI, Google, Mistral. Cost-per-correct-answer as a first-class metric — the number that actually matters for production.

**Regression detection.** Fisher exact test on pass rates, Mann-Whitney U on cost/latency. Catches statistically significant drops between versions. Exit code 1 blocks your PR in CI.

**Local-first.** Data never leaves your machine. No accounts, no SaaS, no telemetry.

---

## Writing tests

YAML:

```yaml
suite: my-agent-tests
agent: my_module.agent       # Python import path
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

Python (for complex assertions):

```python
from agentrial import expect, AgentInput

result = agent(AgentInput(query="Book a flight to Rome"))

e = expect(result).succeeded() \
    .tool_called("search_flights", params_contain={"destination": "FCO"}) \
    .cost_below(0.15) \
    .latency_below(5000)

e.output.contains("confirmed", "Rome")
e.step(0).tool_name("search_flights")

assert e.passed()
```

All assertion types: `output_contains`, `output_contains_any`, `exact_match`, `regex`, `tool_calls` with `params_contain`, per-step expectations via `step_expectations`. See [full assertion docs](https://github.com/alepot55/agentrial/wiki).

---

## Wrapping your agent

agentrial needs a callable: `AgentInput -> AgentOutput`. Native adapters handle the wiring.

### LangGraph

```python
from langgraph.prebuilt import create_react_agent
from agentrial.runner.adapters import wrap_langgraph_agent

graph = create_react_agent(llm, tools=[...])
agent = wrap_langgraph_agent(graph)
```

Automatically captures full trajectory, token usage, real API cost, and execution duration via LangChain callbacks.

### CrewAI

```python
from agentrial.runner.adapters import wrap_crewai_agent
agent = wrap_crewai_agent(crew)
```

### Custom agents

```python
from agentrial.types import AgentInput, AgentOutput, AgentMetadata

def agent(input: AgentInput) -> AgentOutput:
    # Your logic here
    return AgentOutput(
        output="result",
        steps=[],
        metadata=AgentMetadata(total_tokens=100, cost=0.001, duration_ms=500.0),
        success=True,
    )
```

### All supported frameworks

| Framework | Adapter | What it captures |
|---|---|---|
| **LangGraph** | `wrap_langgraph_agent` | Callbacks, trajectory, tokens, cost |
| **CrewAI** | `wrap_crewai_agent` | Task-level trajectory, crew cost |
| **AutoGen** | `wrap_autogen_agent` | v0.4+ and legacy pyautogen |
| **Pydantic AI** | `wrap_pydantic_ai_agent` | Tool calls, response parts, tokens |
| **OpenAI Agents SDK** | `wrap_openai_agents_agent` | Runner integration, tool calls |
| **smolagents (HF)** | `wrap_smolagents_agent` | Dict and object log formats |
| **Any OTel agent** | Automatic | Span capture via OTel SDK |
| **Custom** | `AgentInput -> AgentOutput` | Whatever you return |

---

## CI/CD

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
```

### Regression detection

```yaml
      - run: agentrial compare results.json --baseline baseline.json
```

Fisher exact test (p < 0.05) detects statistically significant regressions. Exit code 1 blocks the PR.

---

## Statistical methods

| Method | Purpose |
|---|---|
| **Wilson score interval** | Pass rate CI — accurate at 0%, 100%, and small N |
| **Bootstrap resampling** | Cost/latency CI — non-parametric, 500 iterations |
| **Fisher exact test** | Regression detection and failure attribution (p < 0.05) |
| **Mann-Whitney U** | Cost/latency comparison between versions |
| **Benjamini-Hochberg** | False discovery rate control for multiple comparisons |

Failure attribution works by grouping trials into pass/fail, comparing tool call distributions at each step, and identifying the step with the lowest p-value as the most likely divergence point.

---

## CLI reference

```bash
agentrial init                              # Scaffold sample project
agentrial run                               # Run all tests
agentrial run tests/ --trials 20            # Custom trials
agentrial run -o results.json               # JSON export
agentrial run --flamegraph                  # Trajectory flame graphs
agentrial run --judge                       # LLM-as-Judge evaluation
agentrial compare results.json -b base.json # Regression detection
agentrial baseline results.json             # Save baseline
agentrial snapshot update / check           # Snapshot testing
agentrial security scan --mcp-config c.json # MCP security scan
agentrial pareto --models m1,m2,m3          # Cost-accuracy Pareto frontier
agentrial prompt track/diff/list            # Prompt version control
agentrial monitor --baseline snap.json      # Production drift detection
```

---

## How it compares

| | agentrial | Promptfoo | LangSmith | DeepEval | Arize Phoenix |
|---|---|---|---|---|---|
| Multi-trial with CI | Free | — | $39/mo | — | — |
| Confidence intervals | Wilson CI | — | — | — | — |
| Step-level failure attribution | Fisher exact | — | — | — | Partial |
| Framework-agnostic | 6 adapters + OTel | Yes | LangChain only | Yes | Yes |
| Cost-per-correct-answer | Yes | — | — | — | — |
| Local-first | Yes | Yes | No | No | Self-host option |

---

## Cost tracking — supported models

Anthropic (Claude 3/3.5/4 family), OpenAI (GPT-4o, 4o-mini, 4 Turbo, 3.5 Turbo), Google (Gemini 1.5 Pro/Flash, 1.0 Pro), Mistral (Large, Medium, Small). 40+ models total. Unknown models get a default estimate with a logged warning.

---

## Contributing

```bash
git clone https://github.com/alepot55/agentrial.git
cd agentrial
pip install -e ".[dev]"
pytest                    # 349 tests
ruff check .
mypy agentrial/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

[MIT](LICENSE)
