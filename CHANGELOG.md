# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0-alpha.1] - 2026-02-06

### Added

- **Trajectory Flame Graphs** (`agentrial.reporters.flamegraph`): SVG visualization of agent
  execution trajectories with collapsible stack frames.
- **Statistical Snapshot Testing** (`agentrial.snapshots`): Golden-file regression testing with
  configurable tolerance via Mann-Whitney U and Fisher exact tests.
- **Calibrated LLM-as-Judge** (`agentrial.evaluators.llm_judge`): Multi-rubric evaluation with
  inter-rater reliability (Krippendorff's alpha) and bias detection.
- **Multi-Agent Evaluation** (`agentrial.evaluators.multi_agent`): Evaluate multi-agent systems
  with per-agent trajectory tracking and interaction metrics.
- **Framework Adapters**: First-class support for 6 agent frameworks:
  - LangGraph (`agentrial.runner.adapters.langgraph`)
  - CrewAI (`agentrial.runner.adapters.crewai`)
  - AutoGen (`agentrial.runner.adapters.autogen`)
  - Pydantic AI (`agentrial.runner.adapters.pydantic_ai`)
  - OpenAI Agents SDK (`agentrial.runner.adapters.openai_agents`)
  - smolagents (`agentrial.runner.adapters.smolagents`)
- **MCP Security Scanner** (`agentrial.security.scanner`): Static analysis for MCP server
  configurations detecting prompt injection, tool shadowing, credential exposure, rug pulls,
  dangerous combinations, and permission escalation.
- **Pytest Plugin** (`agentrial.pytest_plugin`): `@agent_test` decorator for writing agent tests
  with automatic trajectory capture and statistical assertions.
- **Cost-Accuracy Pareto Frontier** (`agentrial.pareto`): Multi-objective optimization analysis
  with dominated-solution pruning and frontier visualization.
- **Prompt Version Control** (`agentrial.prompts`): Git-inspired prompt versioning with diff,
  track, and list operations.
- **Production Drift Detection** (`agentrial.monitor`): Real-time monitoring with CUSUM,
  Page-Hinkley, Kolmogorov-Smirnov, and sliding window detectors.
- **Cloud Dashboard** (`agentrial.dashboard`): FastAPI-based web dashboard for viewing evaluation
  results, with persistent storage and REST API.
- **VS Code Extension** (`vscode-extension/`): TypeScript scaffold for IDE integration.

### Changed

- Renamed package from `agenteval` to `agentrial`.
- `FunctionAdapter` now returns `success=False` when the wrapped function returns `None`.
- `wrap_smolagent` renamed to `wrap_smolagents_agent` (old name kept as alias).
- `snapshot update`, `snapshot check`, and `pareto` CLI commands now accept `--config/-c`.
- Statistical tests (Wilson CI, Bootstrap CI, Fisher exact) bounds tightened for accuracy.

### Fixed

- Scanner no longer crashes on malformed config (non-dict server entries, non-dict tools,
  non-dict env).
- Empty MCP config no longer returns a perfect 10/10 score.
- Prompt versioning handles corrupt JSON gracefully instead of crashing.
- `t_ppf` approximation for large degrees of freedom (df > 120) now returns 1.96 correctly.
- Krippendorff's alpha handles single-item degenerate case with scale-relative consistency.
- LangGraph adapter properly inherits from `BaseAdapter`.
- CUSUM detector respects `min_observations` warmup period.
- `SlidingWindowDetector` uses Fisher exact test for small windows (n < 30).

## [0.1.4] - 2025-12-01

### Added

- Initial release with core evaluation runner, statistical comparisons, and CLI.
- Basic `AgentInput`/`AgentOutput`/`TrajectoryStep` types.
- Wilson score confidence intervals and bootstrap resampling.
- Fisher exact test and Mann-Whitney U test.
- `agentrial run`, `agentrial compare`, `agentrial init` CLI commands.
