# CLAUDE.md — agentrial

## Project Overview

`agentrial` is a statistical evaluation framework for AI agents — "the pytest for agent trajectories." It runs every test N times, computes Wilson confidence intervals for pass rates, and pinpoints which trajectory step diverges between passing and failing runs. Framework-agnostic, local-first, free.

Package: `agentrial` | Python 3.11+ | License: MIT | Version: 0.2.0a1

## Quick Reference

```bash
pip install -e ".[dev]"          # Install for development
pytest --tb=short -q             # Run all tests (~349 tests)
ruff check agentrial/            # Lint
agentrial init                   # Scaffold sample project
agentrial run --trials 5         # Run evaluations
```

## Architecture

```
agentrial/
├── __init__.py                    # Public API: expect, AgentInput, AgentOutput, Suite, etc.
├── types.py                       # Core dataclasses: AgentInput, AgentOutput, TrajectoryStep,
│                                  #   AgentMetadata, ExpectedOutput, TestCase, Suite, EvalResult,
│                                  #   SuiteResult, StepType (enum)
├── cli.py                         # Click CLI entry point (agentrial.cli:main)
├── config.py                      # YAML config loading and test file discovery
├── snapshots.py                   # Statistical snapshot testing (create, save, load, compare)
├── pareto.py                      # Cost-accuracy Pareto frontier analysis
├── prompts.py                     # Prompt version control (PromptStore: track, diff, compare)
├── monitor.py                     # Production drift detection:
│                                  #   CUSUMDetector (with 30-obs warmup)
│                                  #   PageHinkleyDetector
│                                  #   SlidingWindowDetector (Fisher exact for n<30, z-test for n>=30)
│                                  #   KSDetector, DriftDetector (unified)
├── pytest_plugin.py               # @agent_test decorator for pytest integration
│
├── runner/
│   ├── engine.py                  # MultiTrialEngine — orchestrates N trials per test case
│   ├── trajectory.py              # TrajectoryRecorder — captures steps, tokens, cost
│   ├── otel.py                    # OpenTelemetry span capture for any framework
│   └── adapters/
│       ├── __init__.py            # Lazy imports for all wrap_* functions
│       ├── base.py                # BaseAdapter(ABC), FunctionAdapter, wrap_function
│       ├── langgraph.py           # LangGraphAdapter(BaseAdapter) — callbacks + OTel + stream
│       ├── crewai.py              # CrewAIAdapter — crew.kickoff() wrapper
│       ├── autogen.py             # AutoGenAdapter — v0.4+ async and legacy sync
│       ├── pydantic_ai.py         # PydanticAIAdapter — run_sync wrapper
│       ├── openai_agents.py       # OpenAIAgentsAdapter — Runner.run_sync wrapper
│       ├── smolagents.py          # SmolAgentsAdapter — dict and object log formats
│       └── pricing.py             # MODEL_PRICING dict for 40+ LLMs, calculate_cost()
│
├── evaluators/
│   ├── exact.py                   # contains, regex, tool_called, exact_match
│   ├── expect.py                  # Fluent API: expect(result).succeeded().cost_below(0.15)
│   ├── functional.py              # Custom check functions, range checks
│   ├── llm_judge.py               # LLMJudge with Krippendorff's alpha, t-distribution CI,
│   │                              #   gold standard calibration, rule-based fallback
│   ├── multi_agent.py             # Multi-agent eval (delegation, handoff, redundancy)
│   └── step_eval.py               # Per-step and trajectory evaluation
│
├── metrics/
│   ├── basic.py                   # Pass rate, cost, latency, token efficiency
│   ├── statistical.py             # wilson_confidence_interval, bootstrap_confidence_interval,
│   │                              #   fisher_exact_test, mann_whitney_u_test,
│   │                              #   benjamini_hochberg_correction, detect_regression,
│   │                              #   compare_multiple_metrics
│   └── trajectory.py              # Failure attribution via step divergence analysis
│
├── reporters/
│   ├── terminal.py                # Rich terminal tables and formatting
│   ├── json_report.py             # JSON export, load, comparison
│   └── flamegraph.py              # Trajectory flame graphs (terminal + HTML export)
│
├── security/
│   └── scanner.py                 # MCP security scanner: prompt injection, tool shadowing,
│                                  #   permission escalation, dangerous combos, rug pull detection
│
├── dashboard/
│   ├── app.py                     # FastAPI web dashboard
│   ├── models.py                  # Dashboard data models
│   └── store.py                   # Persistent JSON storage backend
│
└── ci/
    └── github_action.py           # GitHub Action YAML generator
```

## Core Concepts

### Agent Interface Contract

An agent is any callable: `AgentInput -> AgentOutput`.

```python
from agentrial.types import AgentInput, AgentOutput, AgentMetadata

def my_agent(input: AgentInput) -> AgentOutput:
    # input.query: str — the question/task
    # input.context: dict[str, Any] — additional context
    result = call_your_llm(input.query)
    return AgentOutput(
        output=result,                        # str — final output
        steps=[],                             # list[TrajectoryStep] — execution trace
        metadata=AgentMetadata(
            total_tokens=150,
            cost=0.002,
            duration_ms=450.0,
        ),
        success=True,
    )
```

Use an adapter for framework-specific agents:
```python
from agentrial.runner.adapters import wrap_langgraph_agent
agent_fn = wrap_langgraph_agent(my_compiled_graph)
```

### TrajectoryStep

```python
@dataclass
class TrajectoryStep:
    step_index: int                          # 0-based
    step_type: StepType                      # TOOL_CALL | LLM_CALL | OBSERVATION | REASONING | OUTPUT
    name: str                                # e.g., "search", "llm_response"
    parameters: dict[str, Any]               # input params
    output: Any                              # step result
    duration_ms: float
    tokens: int
    metadata: dict[str, Any]
```

### Test Case Format (YAML)

```yaml
suite: my-agent-tests
agent: my_module.agent              # Python import path
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
    max_cost: 0.05
    max_latency_ms: 5000
```

### Fluent Assertion API (Python tests)

```python
from agentrial import expect

e = expect(result).succeeded() \
    .tool_called("search_flights", params_contain={"destination": "FCO"}) \
    .cost_below(0.15) \
    .latency_below(5000)

# .output returns OutputExpectation (separate chain)
e.output.contains("confirmed", "Rome")
# .step(i) returns StepExpectation (separate chain)
e.step(0).tool_name("search_flights")

assert e.passed()
```

## CLI Commands

### Core

```
agentrial init                                   # Scaffold sample project
agentrial run [TEST_PATH]                        # Run evaluation suite(s)
  -c, --config PATH                              # Config file path
  -n, --trials INTEGER                           # Trials per test case
  -t, --threshold FLOAT                          # Min pass rate (0-1)
  -o, --output PATH                              # JSON output path
  --json                                         # JSON to stdout
  --flamegraph                                   # Show trajectory flame graphs
  --html PATH                                    # Export flame graph as HTML
  --update-snapshots                             # Save as snapshot baseline
  --judge                                        # Enable LLM-as-Judge evaluation
agentrial compare CURRENT_PATH                   # Regression detection
  -b, --baseline PATH                            # Baseline JSON [required]
agentrial baseline RESULTS_PATH                  # Save as baseline
agentrial config                                 # Show configuration
```

### Snapshots

```
agentrial snapshot update [TEST_PATH]            # Run and save snapshot
  -n, --trials INTEGER
  -c, --config PATH
  -o, --output PATH
agentrial snapshot check [TEST_PATH]             # Compare against snapshot
  -n, --trials INTEGER
  -c, --config PATH
```

### Security

```
agentrial security scan                          # Scan MCP config
  --mcp-config PATH                              # MCP config JSON [required]
  -o, --output PATH                              # Save results as JSON
```

### Pareto Frontier

```
agentrial pareto [TEST_PATH]                     # Cost-accuracy Pareto analysis
  --models TEXT                                  # Comma-separated model names [required]
  -n, --trials INTEGER
  -c, --config PATH
```

### Prompt Version Control

```
agentrial prompt track PROMPT_FILE               # Track new prompt version
  --version TEXT                                 # Explicit version name
  --tag TEXT                                     # Tags in key=value format
agentrial prompt diff VERSION_A VERSION_B        # Diff two versions
agentrial prompt list                            # List all versions
```

### Monitoring

```
agentrial monitor                                # Configure drift monitoring
  --baseline PATH                                # Baseline snapshot JSON [required]
  --dry-run                                      # Show config without running
  --window INTEGER                               # Sliding window size (default: 50)
  --cusum-threshold FLOAT                        # CUSUM threshold (default: 5.0)
```

### Dashboard

```
agentrial dashboard                              # Launch web dashboard
  --port INTEGER                                 # Server port (default: 8000)
  --host TEXT                                    # Server host (default: 127.0.0.1)
  --data-dir PATH                                # Persistence directory
```

## Statistical Methods

| Method | Function | Purpose |
|---|---|---|
| **Wilson score interval** | `wilson_confidence_interval(k, n, confidence)` | Pass rate CI — accurate at boundaries (0%, 100%) and small n |
| **Bootstrap resampling** | `bootstrap_confidence_interval(values, ...)` | CI for cost/latency — 500 iterations, no normality assumption |
| **Fisher exact test** | `fisher_exact_test(k1, n1, k2, n2)` → `(odds_ratio, p_value)` | Regression detection between two runs |
| **Mann-Whitney U** | `mann_whitney_u_test(a, b)` → `(statistic, p_value)` | Non-parametric comparison of cost/latency distributions |
| **Benjamini-Hochberg** | `benjamini_hochberg_correction(p_values, alpha)` | FDR control for multiple comparisons |
| **CUSUM** | `CUSUMDetector(threshold, warmup=30)` | Sequential change detection with warmup period |
| **Page-Hinkley** | `PageHinkleyDetector(delta, threshold)` | Sequential change detection (mean shift) |
| **Sliding window** | `SlidingWindowDetector(window_size, alpha)` | Fisher exact (n<30) or z-test (n>=30) for windowed pass rate |
| **Krippendorff's alpha** | `compute_krippendorff_alpha(ratings)` | Inter-rater reliability for LLM judge (t-distribution CI when M<5) |

**Key implementation notes:**
- `fisher_exact_test()` and `mann_whitney_u_test()` return **tuples** `(statistic, p_value)`, not dicts
- LLM Judge CI uses **t-distribution** `t(df=M-1)` for small repeats (not z=1.96). Warning logged when M<5
- CUSUM has **30-observation warmup** — no alerts before warmup completes
- SlidingWindow uses Fisher exact for **window_size < 30**, z-test for >= 30

## Framework Adapters

| Framework | Import path | Wrap function | Dependency |
|---|---|---|---|
| **LangGraph** | `agentrial.runner.adapters.langgraph` | `wrap_langgraph_agent(graph)` | `langgraph`, `langchain-core` |
| **CrewAI** | `agentrial.runner.adapters.crewai` | `wrap_crewai_agent(crew)` | `crewai` |
| **AutoGen** | `agentrial.runner.adapters.autogen` | `wrap_autogen_agent(agent)` | `autogen-agentchat` |
| **Pydantic AI** | `agentrial.runner.adapters.pydantic_ai` | `wrap_pydantic_ai_agent(agent)` | `pydantic-ai` |
| **OpenAI Agents** | `agentrial.runner.adapters.openai_agents` | `wrap_openai_agent(agent)` | `openai-agents` |
| **smolagents** | `agentrial.runner.adapters.smolagents` | `wrap_smolagents_agent(agent)` | `smolagents` |
| **Custom** | `agentrial.runner.adapters.base` | `wrap_function(fn)` | none |

All adapters lazy-import their framework at runtime. The core package has zero framework dependencies.

## Tech Stack

**Required dependencies** (installed with `pip install agentrial`):
- `click>=8.1.0` — CLI framework
- `pyyaml>=6.0` — YAML config parsing
- `rich>=13.0.0` — Terminal formatting
- `scipy>=1.11.0` — Statistical tests (Fisher, Mann-Whitney, KS)
- `numpy>=1.24.0` — Numerical operations
- `opentelemetry-api>=1.20.0` + `opentelemetry-sdk>=1.20.0` — Span capture

**Optional dependency groups** (`pip install agentrial[group]`):
- `dev` — pytest, pytest-cov, mypy, ruff
- `dashboard` — fastapi, uvicorn
- `judge` — litellm
- `langgraph`, `crewai`, `autogen`, `pydantic-ai`, `openai-agents`, `smolagents` — framework adapters
- `all` — everything

## Coding Standards

- **Formatter/linter**: `ruff` with 100-char line length
- **Type hints**: All public functions annotated. `mypy --strict` target.
- **Imports**: Framework adapters use `_import_<framework>()` pattern — import at runtime only
- **Tests**: pytest, files in `tests/`. Currently 349+ tests.
- **No `\b` before dots** in regex — use `(\.env)` not `\b.env\b`
- **Dataclass returns**: `fisher_exact_test()` and `mann_whitney_u_test()` return tuples, not dicts

## Definition of Done for v0.2.0-alpha.1

- [x] All 349+ tests pass (`pytest --tb=short -q`)
- [x] Package builds (`python -m build` produces .whl and .tar.gz)
- [x] `agentrial --help` shows all commands
- [x] `agentrial init && agentrial run --trials 3` works end-to-end
- [x] 5 CRITICAL, 8 HIGH, 8 MEDIUM bugs from adversarial code review fixed
- [x] README reflects actual API (fluent chain, frameworks, CLI)
- [x] CHANGELOG.md covers all v0.2.0a1 features
- [x] pyproject.toml version = "0.2.0a1"
- [x] LICENSE file present (MIT)
