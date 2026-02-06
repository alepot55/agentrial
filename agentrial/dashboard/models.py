"""Data models for the Cloud Dashboard.

Defines the schema for suite runs, results, alerts, and teams.
These models are database-agnostic — they use dataclasses and can
be serialized to/from JSON or mapped to an ORM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Team:
    """A team of users."""

    id: str
    name: str
    created_at: str = ""
    members: list[str] = field(default_factory=list)


@dataclass
class SuiteRun:
    """A recorded suite execution."""

    id: str
    suite_name: str
    team_id: str
    timestamp: str
    model: str = ""
    prompt_version: str = ""
    overall_pass_rate: float = 0.0
    total_cost: float = 0.0
    total_duration_ms: float = 0.0
    passed: bool = False
    trials: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    case_results: list[CaseRunResult] = field(default_factory=list)


@dataclass
class CaseRunResult:
    """Result for a single test case within a suite run."""

    case_name: str
    pass_rate: float = 0.0
    mean_cost: float = 0.0
    mean_latency_ms: float = 0.0
    trials_passed: int = 0
    trials_total: int = 0
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class AlertConfig:
    """Alert configuration for a suite."""

    id: str
    suite_name: str
    team_id: str
    enabled: bool = True
    pass_rate_threshold: float = 0.8
    cost_threshold: float | None = None
    latency_threshold_ms: float | None = None
    channels: list[AlertChannel] = field(default_factory=list)


@dataclass
class AlertChannel:
    """A notification channel for alerts."""

    type: str  # "slack", "email", "webhook"
    target: str  # URL, email address, or channel name
    enabled: bool = True


@dataclass
class AlertEvent:
    """A triggered alert event."""

    id: str
    alert_config_id: str
    suite_name: str
    team_id: str
    timestamp: str
    severity: str  # "critical", "warning", "info"
    message: str
    metric: str
    baseline_value: float = 0.0
    current_value: float = 0.0
    acknowledged: bool = False


@dataclass
class CostReport:
    """Aggregated cost report."""

    team_id: str
    period_start: str
    period_end: str
    total_cost: float = 0.0
    cost_by_suite: dict[str, float] = field(default_factory=dict)
    cost_by_model: dict[str, float] = field(default_factory=dict)
    total_runs: int = 0
    total_trials: int = 0


@dataclass
class ComparisonView:
    """Side-by-side comparison of two suite runs."""

    run_a: SuiteRun
    run_b: SuiteRun
    pass_rate_delta: float = 0.0
    cost_delta: float = 0.0
    duration_delta: float = 0.0
    improved_cases: list[str] = field(default_factory=list)
    regressed_cases: list[str] = field(default_factory=list)
    unchanged_cases: list[str] = field(default_factory=list)


@dataclass
class TraceStep:
    """A step in a trace for the trace explorer."""

    step_index: int
    step_type: str
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    duration_ms: float = 0.0
    tokens: int = 0
    agent_id: str = ""


@dataclass
class TraceView:
    """Complete trace for the trace explorer."""

    run_id: str
    case_name: str
    trial_index: int
    passed: bool
    steps: list[TraceStep] = field(default_factory=list)
    total_tokens: int = 0
    total_cost: float = 0.0
    duration_ms: float = 0.0
