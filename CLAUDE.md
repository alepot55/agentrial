# CLAUDE.md — AgentEval

## Project Overview

AgentEval is an open-source Python CLI tool for statistically rigorous evaluation of AI agents. It is the "pytest for AI agents": it tests multi-step agent trajectories with automatic multi-trial execution, confidence intervals, and step-level failure attribution.

**Why this exists**: Existing eval tools (LangSmith, Arize Phoenix, Promptfoo, DeepEval, RAGAS) are either LLM-first with agent features bolted on, locked to a specific framework, or lack statistical rigor for non-deterministic systems. No tool today combines trajectory evaluation, multi-trial statistics, and CI/CD integration in a single open-source package.

**Target user**: A developer building AI agents with LangGraph, CrewAI, AutoGen, or similar frameworks, who needs to know if their agent still works after a prompt change, model swap, or code update — before it hits production.

**Business model**: Open-core. Free CLI + GitHub Action. Future paid cloud tier for team dashboards, trace storage, and comparison reports.

---

## Architecture Principles

1. **Agent-native, not LLM-adapted**: Trajectory evaluation and step-level analysis are first-class, not afterthoughts
2. **Framework-agnostic via OpenTelemetry**: OTel spans as universal trace format; framework-specific adapters are thin wrappers
3. **Statistical rigor by default**: Every evaluation runs N trials with confidence intervals. No single-run pass/fail
4. **Cost-aware**: Track token usage and API costs per run. Cost-per-correct-answer as a first-class metric
5. **Local-first**: All evaluation runs locally. Prompts and data never leave the user's machine
6. **CLI-first**: The primary interface is the terminal. No web UI in MVP

---

## Tech Stack

- **Language**: Python 3.11+ (ecosystem compatibility with all agent frameworks)
- **Package manager**: `uv` (fast, modern)
- **CLI framework**: `click` or `typer`
- **Statistics**: `scipy.stats` for confidence intervals, bootstrap resampling; `numpy` for numerical ops
- **OTel integration**: `opentelemetry-api`, `opentelemetry-sdk` for trace ingestion
- **Output format**: Rich terminal tables via `rich`; JSON export for CI/CD
- **Testing**: `pytest` for internal tests
- **Packaging**: `pyproject.toml` with `hatchling` or `setuptools`

**Do NOT use**:
- Heavy web frameworks (Flask, FastAPI) — not needed for MVP
- Databases — file-based storage (JSON/JSONL) for traces in MVP
- Frontend/React — no web UI in MVP
- Docker — not needed for local CLI tool
- LangChain as a dependency — we are framework-agnostic

---

## MVP Scope (Weeks 1-4)

### Week 1-2: Core engine
- [ ] Project scaffolding: pyproject.toml, CLI entry point, directory structure
- [ ] **Test case definition format**: YAML/Python files that define agent test scenarios
- [ ] **Agent runner**: Execute an agent function N times, capture full trajectory (steps, tool calls, outputs, timing, token usage)
- [ ] **Multi-trial engine**: Run N trials (default 10), collect all trajectories
- [ ] **Basic metrics**: task_completion (binary), cost_per_run, latency, token_count
- [ ] **Statistical module**: Mean pass rate, 95% CI via Wilson score interval, bootstrap resampling (500 iterations)
- [ ] **CLI output**: `agenteval run` prints a rich table with pass rate, CI, cost, latency

### Week 3: Step-level analysis
- [ ] **Trajectory data model**: Ordered list of steps, each with: action_type, tool_name, parameters, output, duration, tokens
- [ ] **Step-level pass/fail**: Per-step evaluation against expected behavior (exact match, contains, regex, custom function)
- [ ] **Failure attribution**: When task fails, identify which step diverged from successful runs
- [ ] **Divergence detection**: Compare action distributions at each step between successful and failed runs (chi-squared test or Fisher exact test)

### Week 4: CI/CD + DX
- [ ] **GitHub Action**: YAML template that runs `agenteval run` on PR, posts summary as comment, fails if pass rate < threshold
- [ ] **JSON report export**: Machine-readable output for CI/CD integration
- [ ] **Config file**: `agenteval.yml` for project-level defaults (trials, threshold, model, framework)
- [ ] **LangGraph adapter**: First framework integration via OTel spans
- [ ] **README + quickstart guide**: 5-minute setup, copy-paste examples

### Deferred (post-MVP)
- CrewAI, AutoGen, Pydantic AI adapters
- LLM-as-judge evaluators
- Pareto frontier visualization
- Cloud dashboard
- MCP security scanning
- Prompt version control

---

## Directory Structure

```
agenteval/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── agenteval/
│   ├── __init__.py
│   ├── cli.py                  # CLI entry point (click/typer)
│   ├── config.py               # YAML config loading
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── engine.py           # Multi-trial execution engine
│   │   ├── trajectory.py       # Trajectory data model
│   │   └── adapters/
│   │       ├── __init__.py
│   │       ├── base.py         # Abstract adapter interface
│   │       └── langgraph.py    # LangGraph OTel adapter
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── basic.py            # task_completion, cost, latency, tokens
│   │   ├── statistical.py      # CI, bootstrap, significance tests
│   │   └── trajectory.py       # Step-level metrics, divergence detection
│   ├── evaluators/
│   │   ├── __init__.py
│   │   ├── exact.py            # Exact match, contains, regex
│   │   ├── functional.py       # Custom Python function evaluators
│   │   └── step_eval.py        # Per-step evaluation logic
│   ├── reporters/
│   │   ├── __init__.py
│   │   ├── terminal.py         # Rich table output
│   │   └── json_report.py      # JSON export for CI/CD
│   └── ci/
│       ├── __init__.py
│       └── github_action.py    # GitHub Action helper
├── tests/
│   ├── test_engine.py
│   ├── test_metrics.py
│   ├── test_statistical.py
│   └── test_evaluators.py
├── examples/
│   ├── simple_agent/           # Minimal working example
│   └── langgraph_agent/        # LangGraph integration example
└── .github/
    └── workflows/
        └── agenteval.yml       # Dogfood: test ourselves with ourselves
```

---

## Test Case Format

Users define test cases in YAML or Python. YAML example:

```yaml
# tests/test_flight_search.yml
suite: flight-search-agent
agent: my_agent.flight_search  # Python import path to agent function
trials: 10
threshold: 0.85  # Minimum pass rate to pass CI

cases:
  - name: cheapest-flight-basic
    input:
      query: "Find the cheapest flight from Rome to Tokyo in March"
    expected:
      output_contains: ["cheapest", "flight"]
      tool_calls:
        - tool: search_flights
          params_contain: {origin: "FCO", destination: "TYO"}
    steps:
      - name: tool-selection
        expected_tool: search_flights
      - name: result-parsing
        check: output.price == min(tool_results.prices)

  - name: no-flights-available
    input:
      query: "Find flights from Rome to Atlantis"
    expected:
      output_contains: ["no flights", "not available"]
```

Python example (more flexible):

```python
# tests/test_flight_search.py
from agenteval import TestCase, Suite, expect

suite = Suite(
    name="flight-search",
    agent="my_agent.flight_search",
    trials=10,
    threshold=0.85,
)

@suite.case
def test_cheapest_flight(result):
    expect(result).tool_called("search_flights")
    expect(result).step(2).params_contain(origin="FCO")
    expect(result).output.contains("cheapest")
    expect(result).cost_below(0.15)  # Max $0.15 per run
```

---

## Agent Interface Contract

AgentEval needs a standard way to call any agent. The agent must expose a callable with this signature:

```python
from agenteval.types import AgentInput, AgentOutput, TrajectoryStep

def my_agent(input: AgentInput) -> AgentOutput:
    """
    AgentInput: {query: str, context: dict}
    AgentOutput: {
        output: str,
        steps: list[TrajectoryStep],
        metadata: {tokens: int, cost: float, duration: float}
    }
    """
```

For framework-specific adapters, we wrap the framework's native agent:

```python
from agenteval.adapters.langgraph import wrap_langgraph_agent
from my_app import graph  # User's LangGraph graph

agent = wrap_langgraph_agent(graph)
```

---

## Statistical Methods

### Pass Rate with Confidence Interval
- Use **Wilson score interval** (not naive proportion) for pass/fail rates with small N
- Default N=10 trials, report 95% CI
- Example output: "Pass rate: 72% (95% CI: 55-84%)"

### Bootstrap Resampling
- For cost and latency distributions: 500 bootstrap iterations
- Report median and 95% percentile interval

### Regression Detection
- Compare current run against baseline (previous saved run)
- Use **Fisher exact test** for pass rate comparison (small samples)
- Use **Mann-Whitney U** for cost/latency comparison (non-parametric)
- Flag as regression if p < 0.05

### Divergence Detection for Failure Attribution
- At each trajectory step, compute action distribution for successful vs failed runs
- Use **Fisher exact test** (2x2: success/fail × action_A/action_B) for each step
- Step with lowest p-value is the most likely failure point
- Report as: "Step 3 (tool_selection) shows significant divergence between successful and failed runs (p=0.003)"

---

## Coding Standards

- **Type hints everywhere**: All function signatures must have complete type annotations
- **Dataclasses or Pydantic models** for data structures (TrajectoryStep, AgentOutput, EvalResult, etc.)
- **No classes where functions suffice**: Prefer functional style for metrics and evaluators
- **Docstrings**: Google style, required for all public functions
- **Error handling**: Fail fast with clear messages. Never silently swallow errors
- **No print()**: Use `rich.console` for terminal output, `logging` for debug
- **Tests**: Every module must have corresponding tests. Use pytest fixtures
- **Dependencies**: Minimize. Every new dependency needs justification. Core functionality must work with minimal deps

---

## CLI Commands (MVP)

```bash
# Run evaluation suite
agenteval run [--config agenteval.yml] [--trials 10] [--threshold 0.85]

# Run a single test file
agenteval run tests/test_flight_search.yml

# Compare current results against a saved baseline
agenteval compare --baseline results/baseline.json

# Save current results as new baseline
agenteval baseline --save

# Show configuration
agenteval config
```

---

## Key Design Decisions

1. **Why CLI-first, not web UI?** Target user is a developer in their terminal/IDE. Web UIs add complexity and deployment burden. The CLI runs locally, produces actionable output, and integrates with CI/CD natively.

2. **Why N=10 default trials?** Balances statistical power with cost. At N=10, a Wilson 95% CI for 70% pass rate is ±14pp — enough to detect meaningful regressions. Users can increase for tighter bounds.

3. **Why Wilson score, not normal approximation?** Wilson is accurate for small N and extreme proportions (0% or 100%). Normal approximation can produce impossible CIs (<0% or >100%) with small samples.

4. **Why OTel for framework integration?** LangGraph, Pydantic AI, OpenAI SDK, and Semantic Kernel already emit OTel spans. Building on OTel means we get these integrations nearly for free, without maintaining per-framework instrumentation code.

5. **Why file-based storage, not a database?** MVP simplicity. JSON/JSONL files are human-readable, git-friendly, and require zero setup. Database is a post-MVP optimization when trace volume justifies it.

6. **Why LangGraph first?** 4.2M monthly PyPI downloads, largest user base among agent frameworks. Maximizes early adoption potential.

---

## Content Marketing Plan (for context, not for Claude Code)

This section exists so Claude Code understands the broader strategy when writing README, examples, and documentation:

- **Launch post**: "I tested 5 coding agents 10 times each. Here's why benchmarks lie." — with real data generated by AgentEval
- **Target communities**: Hacker News, r/MachineLearning, r/LocalLLaMA, LangGraph Discord, Twitter/X AI community
- **README tone**: Technical but approachable. Lead with the problem ("Your agent passes Monday, fails Wednesday. Same prompt."), then the solution (3-line install + run), then the depth (statistical methods, trajectory analysis)
- **Examples must be copy-pasteable**: A developer should go from `pip install agenteval` to seeing their first eval report in under 5 minutes

---

## What NOT to Build

- No web dashboard (post-MVP)
- No user accounts or auth
- No cloud storage or remote APIs
- No LLM-as-judge in MVP (add later, avoid dependency on paid APIs for core functionality)
- No support for non-Python agents (post-MVP)
- No custom visualization (Rich tables are enough for MVP)
- No plugin system (premature abstraction)

---

## Definition of Done for MVP

The MVP is ready to ship (as a v0.1.0 on PyPI and a blog post) when:

1. `pip install agenteval` works
2. A user can define a test suite in YAML or Python
3. `agenteval run` executes N trials and prints pass rate with 95% CI, cost, and latency
4. Step-level failure attribution identifies which step caused failures
5. A GitHub Action template blocks PRs when pass rate drops below threshold
6. A working example with a simple LangGraph agent is included
7. README explains the tool in <2 minutes of reading
8. The project dogfoods itself (our own CI runs agenteval)
