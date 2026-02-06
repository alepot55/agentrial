<p align="center">
  <h1 align="center">agentrial</h1>
  <p align="center">
    <strong>The pytest for AI agents. Run your agent 100 times, get confidence intervals instead of anecdotes.</strong>
  </p>
  <p align="center">
    <a href="https://pypi.org/project/agentrial/"><img alt="PyPI" src="https://img.shields.io/pypi/v/agentrial?color=blue"></a>
    <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11+-blue.svg"></a>
    <a href="https://github.com/alepot55/agentrial/actions"><img alt="Tests" src="https://img.shields.io/badge/tests-438%20passed-brightgreen"></a>
    <a href="https://marketplace.visualstudio.com/items?itemName=alepot55.agentrial"><img alt="VS Code" src="https://img.shields.io/badge/VS%20Code-marketplace-blue"></a>
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
╰───────────────────────────────────────────────────────── Threshold: 80% ─╯
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

## Why this exists

Every agent framework ships with benchmarks showing 90%+ accuracy. But run those same agents 100 times on the same task, and you'll see pass rates drop to 60-80% with wide variance. The benchmarks measure one run; production sees thousands.

No existing tool gives you statistically rigorous, framework-agnostic agent testing that runs in CI/CD. LangSmith requires a paid account and locks you to LangChain. Promptfoo doesn't do multi-trial with confidence intervals. DeepEval and Arize don't do trajectory-level failure attribution. agentrial fills that gap: open-source, free, local-first, works with any agent framework.

---

## What it does

- **Multi-trial execution** — Run every test N times automatically. A single pass means nothing for non-deterministic agents.
- **Wilson confidence intervals** — Statistically accurate pass rates, even with small samples and extreme proportions (0% or 100%).
- **Step-level failure attribution** — Pinpoints *which tool call* diverges between passing and failing runs using Fisher exact test.
- **Real cost tracking** — Actual API costs from model metadata, 45+ models supported across Anthropic, OpenAI, Google, Mistral, Meta, DeepSeek.
- **Regression detection** — Fisher exact test catches reliability drops between versions. Blocks PRs in CI when quality degrades.
- **Local-first** — Your data never leaves your machine. No accounts, no SaaS, no telemetry.
- **Agent Reliability Score** — A single 0-100 composite metric that combines accuracy, consistency, cost efficiency, latency, trajectory quality, and failure recovery. Weighted scoring with transparent breakdown — one number to track across releases.
- **Production monitoring** — Deploy `agentrial monitor` as a cron job or sidecar. CUSUM and Page-Hinkley detectors catch drift in pass rate, cost, and latency. Kolmogorov-Smirnov test detects distribution shifts. Alerts before users notice.

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
agentrial run tests/          # Finds test_*.yml, test_*.yaml
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

e = expect(result).succeeded() \
    .tool_called("search_flights", params_contain={"destination": "FCO"}) \
    .cost_below(0.15) \
    .latency_below(5000)

# Output checks return OutputExpectation (separate chain)
e.output.contains("confirmed", "Rome")

# Step checks return StepExpectation (separate chain)
e.step(0).tool_name("search_flights").params_contain(destination="FCO")

assert e.passed()
```

| Method | Description |
|---|---|
| `.succeeded()` | Agent completed without error |
| `.output.contains(*strings)` | Output contains all substrings |
| `.output.equals(string)` | Exact match |
| `.output.matches(regex)` | Regex match |
| `.tool_called(name, params_contain={})` | Tool was called with params |
| `.step(i).tool_name(name)` | Step i called named tool |
| `.step(i).params_contain(**kw)` | Step i had params matching kw |
| `.cost_below(max_usd)` | Cost under threshold |
| `.latency_below(max_ms)` | Latency under threshold |
| `.tokens_below(max_tokens)` | Tokens under threshold |
| `.trajectory_length(min, max)` | Step count within bounds |
| `.passed()` | Returns `True` if all pass |
| `.get_failures()` | Returns failure messages |

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

## Advanced features

### Trajectory flame graphs

Visualize agent execution paths across trials. Identify where passing and failing runs diverge.

```bash
agentrial run --flamegraph          # Terminal visualization
agentrial run --flamegraph --html flamegraph.html  # Interactive HTML export
```

### LLM-as-Judge

Use a second LLM to evaluate response quality with calibrated scoring.

```bash
agentrial run --judge               # Add judge evaluation
```

Implements Krippendorff's alpha for inter-rater reliability and t-distribution CI for score estimates. Calibration protocol ensures judge consistency before scoring.

### Snapshot testing

Capture baseline behavior and detect regressions automatically.

```bash
agentrial snapshot update           # Save current behavior as baseline
agentrial snapshot check            # Compare against baseline
```

Uses Fisher exact test on pass rates and Mann-Whitney U on cost/latency, with Benjamini-Hochberg correction across all comparisons.

### MCP security scanner

Audit MCP server configurations for 6 vulnerability classes: prompt injection, tool shadowing, data exfiltration, permission escalation, rug pull, and configuration weakness.

```bash
agentrial security scan --mcp-config servers.json
```

### Multi-agent evaluation

Evaluate multi-agent systems with delegation accuracy, handoff fidelity, redundancy rate, and cascade failure metrics.

### Pareto frontier analysis

Find the optimal cost-accuracy trade-off across models.

```bash
agentrial pareto --models claude-3-haiku,gpt-4o-mini,gemini-flash
```

### Prompt version control

Track, diff, and manage prompt versions with statistical comparison between versions.

```bash
agentrial prompt track prompts/v2.txt
agentrial prompt diff v1 v2
agentrial prompt list
```

### Agent Reliability Score

A composite 0-100 metric combining 6 weighted components: accuracy (40%), consistency (20%), cost efficiency (10%), latency (10%), trajectory quality (10%), and recovery (10%).

```bash
agentrial ars results.json
agentrial ars results.json --cost-ceiling 0.5
```

### Benchmark registry

Publish evaluation results as verifiable, shareable benchmark files with SHA-256 integrity checksums.

```bash
agentrial publish results.json --agent-name my-agent --agent-version 1.0.0
agentrial verify --agent-name my-agent --agent-version 1.0.0 --suite-name my-suite
```

### Eval packs

Domain-specific evaluation packages distributed as Python packages via entry points. Install a pack, get specialized test suites and evaluators.

```bash
agentrial packs list               # Show installed packs
```

### Dashboard

Local FastAPI dashboard for browsing results, comparing runs, and tracking trends.

```bash
agentrial dashboard                # Start at http://localhost:8080
```

---

## VS Code extension

Browse test suites, run evaluations, view flame graphs, and compare snapshots — all from your editor.

Install from the [VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=alepot55.agentrial) or search "agentrial" in VS Code extensions.

Features:
- Suite explorer sidebar with test case tree
- Run suites and individual test cases with one click
- Interactive trajectory flame graph visualization
- Snapshot comparison for regression detection
- MCP security scan integration
- Auto-refresh on YAML file changes

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
| **CUSUM / Page-Hinkley** | Sequential change-point detection for production monitoring |
| **Kolmogorov-Smirnov** | Distribution shift detection for cost and latency |
| **Krippendorff's alpha** | Inter-rater reliability for LLM-as-Judge with t-distribution CI |

### Failure attribution

When tests fail, agentrial analyzes trajectory divergence:
1. Groups trials by pass/fail
2. At each step, compares distribution of tool calls
3. Fisher exact test identifies the step with significant divergence
4. Reports the divergent step with a recommendation

---

## CLI Reference

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
agentrial publish results.json --agent-name me --agent-version 1.0  # Publish benchmark
agentrial verify --agent-name me --agent-version 1.0 --suite-name s # Verify integrity
agentrial packs list                        # List installed eval packs
agentrial dashboard                         # Start local dashboard
agentrial config                            # Show configuration
```

| Flag | Short | Description | Default |
|---|---|---|---|
| `--config` | `-c` | Config file path | `agentrial.yml` |
| `--trials` | `-n` | Trials per test case | `10` |
| `--threshold` | `-t` | Min pass rate (0-1) | `0.85` |
| `--output` | `-o` | JSON output path | — |
| `--json` | | JSON to stdout | `false` |
| `--flamegraph` | | Show trajectory flame graphs | `false` |
| `--html` | | Export flame graph HTML | — |
| `--judge` | | Enable LLM-as-Judge | `false` |
| `--update-snapshots` | | Save as snapshot baseline | `false` |

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

## Supported Frameworks

| Framework | Status | Notes |
|---|---|---|
| **LangGraph** | Native adapter | Full trajectory, callbacks, token tracking |
| **CrewAI** | Native adapter | Task-level trajectory, crew cost tracking |
| **AutoGen** | Native adapter | v0.4+ (autogen-agentchat) and legacy pyautogen |
| **Pydantic AI** | Native adapter | Tool calls, response parts, token usage |
| **OpenAI Agents SDK** | Native adapter | Runner integration, tool call capture |
| **smolagents (HF)** | Native adapter | Dict and object log formats |
| **Any OTel-instrumented agent** | Supported | Automatic span capture via OTel SDK |
| **Custom** | Supported | `AgentInput -> AgentOutput` protocol |

## Supported Models (cost tracking)

| Provider | Models |
|---|---|
| **Anthropic** | Claude 3 Haiku/Sonnet/Opus, Claude 3.5, Claude Sonnet 4.5, Claude Opus 4 |
| **OpenAI** | GPT-4o, GPT-4o-mini, GPT-4 Turbo, GPT-3.5 Turbo, o1, o3-mini |
| **Google** | Gemini 2.0 Flash, Gemini 1.5 Pro/Flash, Gemini 1.0 Pro |
| **Mistral** | Large, Medium, Small, Codestral, Pixtral |
| **Meta** | Llama 3.3 70B, Llama 3.1 405B/70B |
| **DeepSeek** | DeepSeek Chat, DeepSeek Reasoner |

---

## Contributing

```bash
git clone https://github.com/alepot55/agentrial.git
cd agentrial
pip install -e ".[dev]"
pytest                    # 438 tests
ruff check .
mypy agentrial/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

[MIT](LICENSE)
