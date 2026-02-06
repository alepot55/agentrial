# agentrial for VS Code

Statistical evaluation of AI agents, directly in your editor.

## Features

- **Suite Explorer**: Browse test suites and cases from the sidebar
- **Inline Results**: See pass rates, confidence intervals, and costs for each test
- **Trajectory Flame Graphs**: Interactive visualization of agent execution paths
- **Snapshot Comparison**: Detect regressions between versions
- **MCP Security Scan**: Run security analysis on MCP configurations
- **Auto-refresh**: Results update automatically when test files change

## Quick Start

1. Install [agentrial](https://pypi.org/project/agentrial/): `pip install agentrial`
2. Open a project with an `agentrial.yml` config file
3. The extension activates automatically

## Commands

| Command | Description |
|---------|-------------|
| `Agentrial: Run Suite` | Run all tests in the current suite |
| `Agentrial: Run Test Case` | Run a single test case |
| `Agentrial: Show Flame Graph` | Open trajectory flame graph |
| `Agentrial: Compare Snapshot` | Compare current results against baseline |
| `Agentrial: Security Scan` | Run MCP security analysis |

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `agentrial.pythonPath` | `python` | Path to Python interpreter |
| `agentrial.defaultTrials` | `10` | Default number of trials per test |
| `agentrial.defaultThreshold` | `0.85` | Pass rate threshold (0-1) |
| `agentrial.autoRefresh` | `true` | Auto-refresh results on file change |

## Requirements

- Python 3.11+
- agentrial CLI (`pip install agentrial`)

## Links

- [GitHub](https://github.com/alepot55/agentrial)
- [PyPI](https://pypi.org/project/agentrial/)
