<p align="center">
  <h1 align="center">agentrial</h1>
  <p align="center">
    <strong>The pytest for AI agents. Run your agent 100 times, get confidence intervals instead of anecdotes.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/agentrial/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agentrial?color=blue"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
    <a href="https://github.com/alepot55/agentrial/actions"><img alt="Tests" src="https://img.shields.io/badge/tests-450%20passed-brightgreen"></a>
    <a href="https://marketplace.visualstudio.com/items?itemName=alepot55.agentrial-vscode"><img alt="VS Code" src="https://img.shields.io/badge/VS%20Code-marketplace-blue"></a>
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
╭──────────────────────────────────────────────────────────────────────────╮
│ my-agent - FAILED                                                        │
╰───────────────────────────────────────────────────────── Threshold: 85% ─╯
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

**Real cost tracking.** Token usage from API response metadata, not estimates. 45+ models across Anthropic, OpenAI, Google, Mistral, Meta, DeepSeek. Cost-per-correct-answer as a first-class metric — the number that actually matters for production.

**Regression detection.** Fisher exact test on pass rates, Mann-Whitney U on cost/latency. Catches statistically significant drops between versions. Exit code 1 blocks your PR in CI.

**Agent Reliability Score.** A single 0-100 composite metric that combines accuracy (40%), consistency (20%), cost efficiency (10%), latency (10%), trajectory quality (10%), and recovery (10%). One number to track across releases — like Lighthouse for agents.

**Production monitoring.** Deploy `agentrial monitor` as a cron job or sidecar. CUSUM and Page-Hinkley detectors catch drift in pass rate, cost, and latency. Kolmogorov-Smirnov test detects distribution shifts. Alerts before users notice.

**Local-first.** Data never leaves your machine. No accounts, no SaaS, no telemetry.

---

## Writing tests

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

For complex assertions, use the fluent Python API:

```python
from agentrial import expect, AgentInput

result = agent(AgentInput(query="Book a flight to Rome"))

expect(result).succeeded() \
    .tool_called("search_flights", params_contain={"destination": "FCO"}) \
    .cost_below(0.15) \
    .latency_below(5000)
```

All assertion types: `output_contains`, `output_contains_any`, `exact_match`, `regex`, `tool_calls` with `params_contain`, per-step expectations via `step_expectations`. See [full docs](https://github.com/alepot55/agentrial/wiki).

---

## Wrapping your agent

agentrial needs a callable: `AgentInput -> AgentOutput`. Native adapters handle the wiring.

```python
# LangGraph
from agentrial.runner.adapters import wrap_langgraph_agent
agent = wrap_langgraph_agent(your_compiled_graph)

# CrewAI
from agentrial.runner.adapters import wrap_crewai_agent
agent = wrap_crewai_agent(crew)

# Custom — implement the protocol directly
from agentrial.types import AgentInput, AgentOutput, AgentMetadata

def agent(input: AgentInput) -> AgentOutput:
    return AgentOutput(
        output="result", steps=[],
        metadata=AgentMetadata(total_tokens=100, cost=0.001, duration_ms=500.0),
        success=True,
    )
```

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

Regression detection between runs:

```bash
agentrial compare results.json --baseline baseline.json
```

Fisher exact test (p < 0.05) detects statistically significant regressions. Exit code 1 blocks the PR.

---

## Advanced features

### Trajectory flame graphs

```bash
agentrial run --flamegraph                         # Terminal
agentrial run --flamegraph --html flamegraph.html   # Interactive HTML
```

Visualize agent execution paths across trials. See where passing and failing runs diverge, step by step.

### LLM-as-Judge

```bash
agentrial run --judge
```

A second LLM evaluates response quality with calibrated scoring. Krippendorff's alpha for inter-rater reliability, t-distribution CI for score estimates. Calibration protocol runs before scoring to ensure consistency.

### Snapshot testing

```bash
agentrial snapshot update     # Save current behavior as baseline
agentrial snapshot check      # Compare against baseline
```

Fisher exact test on pass rates, Mann-Whitney U on cost/latency, Benjamini-Hochberg correction across all comparisons.

### MCP security scanner

```bash
agentrial security scan --mcp-config servers.json
```

Audits MCP server configurations for 6 vulnerability classes: prompt injection, tool shadowing, data exfiltration, permission escalation, rug pull, configuration weakness.

### Pareto frontier

```bash
agentrial pareto --models claude-3-haiku,gpt-4o-mini,gemini-flash
```

Find the optimal cost-accuracy trade-off across models. ASCII plot in terminal.

### Prompt version control

```bash
agentrial prompt track prompts/v2.txt
agentrial prompt diff v1 v2
```

Track, diff, and compare prompt versions with statistical significance testing between them.

### Benchmark registry

```bash
agentrial publish results.json --agent-name my-agent --agent-version 1.0.0
agentrial verify --agent-name my-agent --agent-version 1.0.0 --suite-name my-suite
```

Publish evaluation results as verifiable benchmark files with SHA-256 integrity checksums.

### Multi-agent evaluation

Delegation accuracy, handoff fidelity, redundancy rate, cascade failure depth, communication efficiency — five metrics for multi-agent systems.

### Dashboard

```bash
agentrial dashboard
```

Local FastAPI dashboard for browsing results, comparing runs, and tracking trends.

### Eval packs

```bash
agentrial packs list
```

Domain-specific evaluation packages via Python entry points. Install a pack, get specialized test templates and evaluators.

---

## VS Code extension

Browse test suites, run evaluations, view flame graphs, and compare snapshots from your editor. Install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=alepot55.agentrial-vscode) or search "agentrial" in extensions.

---

## Statistical methods

| Method | Purpose |
|---|---|
| **Wilson score interval** | Pass rate CI — accurate at 0%, 100%, and small N |
| **Bootstrap resampling** | Cost/latency CI — non-parametric, 500 iterations |
| **Fisher exact test** | Regression detection and failure attribution (p < 0.05) |
| **Mann-Whitney U** | Cost/latency comparison between versions |
| **Benjamini-Hochberg** | False discovery rate control for multiple comparisons |
| **CUSUM / Page-Hinkley** | Sequential change-point detection for production monitoring |
| **Kolmogorov-Smirnov** | Distribution shift detection |
| **Krippendorff's alpha** | Inter-rater reliability for LLM-as-Judge |

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
agentrial ars results.json                  # Agent Reliability Score
agentrial publish results.json --agent-name X --agent-version Y
agentrial verify --agent-name X --agent-version Y --suite-name Z
agentrial packs list                        # Installed eval packs
agentrial dashboard                         # Local dashboard
```

---

## How it compares

| | agentrial | Promptfoo | LangSmith | DeepEval | Arize Phoenix |
|---|---|---|---|---|---|
| Multi-trial with CI | **Free** | — | $39/mo | — | — |
| Confidence intervals | Wilson CI | — | — | — | — |
| Step-level failure attribution | Fisher exact | — | — | — | Partial |
| Framework-agnostic | 6 adapters + OTel | Yes | LangChain only | Yes | Yes |
| Cost-per-correct-answer | Yes | — | — | — | — |
| LLM-as-Judge with calibration | Krippendorff α | — | Yes | Yes | — |
| Composite reliability score | ARS (0-100) | — | — | — | — |
| MCP security scanning | 6 vuln classes | — | — | — | — |
| Production drift detection | CUSUM + PH + KS | — | — | — | Partial |
| VS Code extension | Yes | — | — | — | — |
| Local-first | Yes | Yes | No | No | Self-host option |

---

## Contributing

```bash
git clone https://github.com/alepot55/agentrial.git
cd agentrial
pip install -e ".[dev]"
pytest                    # 450 tests
ruff check .
mypy agentrial/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

[MIT](LICENSE)