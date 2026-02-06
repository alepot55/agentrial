"""In-memory data store for the dashboard.

Provides a storage layer that can be backed by JSON files (dev)
or a database (production). This implementation uses in-memory
dicts with optional JSON persistence.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentrial.dashboard.models import (
    AlertConfig,
    AlertEvent,
    CaseRunResult,
    ComparisonView,
    CostReport,
    SuiteRun,
    Team,
)


class DashboardStore:
    """In-memory store with optional file persistence."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        """Initialize the store.

        Args:
            data_dir: Optional directory for JSON file persistence.
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self._teams: dict[str, Team] = {}
        self._runs: dict[str, SuiteRun] = {}
        self._alerts: dict[str, AlertConfig] = {}
        self._alert_events: dict[str, AlertEvent] = {}

        if self.data_dir:
            self._load()

    # --- Teams ---

    def create_team(self, name: str, members: list[str] | None = None) -> Team:
        """Create a new team."""
        team = Team(
            id=_gen_id(),
            name=name,
            created_at=_now(),
            members=members or [],
        )
        self._teams[team.id] = team
        self._persist()
        return team

    def get_team(self, team_id: str) -> Team | None:
        return self._teams.get(team_id)

    def list_teams(self) -> list[Team]:
        return list(self._teams.values())

    # --- Suite Runs ---

    def record_run(
        self,
        suite_name: str,
        team_id: str,
        overall_pass_rate: float,
        total_cost: float,
        total_duration_ms: float,
        passed: bool,
        trials: int = 0,
        model: str = "",
        prompt_version: str = "",
        case_results: list[CaseRunResult] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SuiteRun:
        """Record a suite run."""
        run = SuiteRun(
            id=_gen_id(),
            suite_name=suite_name,
            team_id=team_id,
            timestamp=_now(),
            model=model,
            prompt_version=prompt_version,
            overall_pass_rate=overall_pass_rate,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
            passed=passed,
            trials=trials,
            metadata=metadata or {},
            case_results=case_results or [],
        )
        self._runs[run.id] = run
        self._persist()
        return run

    def get_run(self, run_id: str) -> SuiteRun | None:
        return self._runs.get(run_id)

    def list_runs(
        self,
        team_id: str | None = None,
        suite_name: str | None = None,
        limit: int = 100,
    ) -> list[SuiteRun]:
        """List runs with optional filters."""
        runs = list(self._runs.values())
        if team_id:
            runs = [r for r in runs if r.team_id == team_id]
        if suite_name:
            runs = [r for r in runs if r.suite_name == suite_name]
        runs.sort(key=lambda r: r.timestamp, reverse=True)
        return runs[:limit]

    def get_trend(
        self,
        suite_name: str,
        team_id: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Get pass rate trend for a suite over time."""
        runs = self.list_runs(team_id=team_id, suite_name=suite_name, limit=limit)
        return [
            {
                "timestamp": r.timestamp,
                "pass_rate": r.overall_pass_rate,
                "cost": r.total_cost,
                "model": r.model,
                "passed": r.passed,
            }
            for r in reversed(runs)  # Chronological order
        ]

    # --- Comparisons ---

    def compare_runs(self, run_id_a: str, run_id_b: str) -> ComparisonView | None:
        """Compare two suite runs side by side."""
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)
        if not run_a or not run_b:
            return None

        pass_rate_delta = run_b.overall_pass_rate - run_a.overall_pass_rate
        cost_delta = run_b.total_cost - run_a.total_cost
        duration_delta = run_b.total_duration_ms - run_a.total_duration_ms

        # Compare case results
        cases_a = {c.case_name: c for c in run_a.case_results}
        cases_b = {c.case_name: c for c in run_b.case_results}
        all_cases = set(cases_a.keys()) | set(cases_b.keys())

        improved = []
        regressed = []
        unchanged = []

        for case_name in all_cases:
            ca = cases_a.get(case_name)
            cb = cases_b.get(case_name)
            if ca and cb:
                delta = cb.pass_rate - ca.pass_rate
                if delta > 0.05:
                    improved.append(case_name)
                elif delta < -0.05:
                    regressed.append(case_name)
                else:
                    unchanged.append(case_name)
            elif cb:
                improved.append(case_name)  # New case
            else:
                regressed.append(case_name)  # Removed case

        return ComparisonView(
            run_a=run_a,
            run_b=run_b,
            pass_rate_delta=pass_rate_delta,
            cost_delta=cost_delta,
            duration_delta=duration_delta,
            improved_cases=improved,
            regressed_cases=regressed,
            unchanged_cases=unchanged,
        )

    # --- Alerts ---

    def create_alert(
        self,
        suite_name: str,
        team_id: str,
        pass_rate_threshold: float = 0.8,
        cost_threshold: float | None = None,
        latency_threshold_ms: float | None = None,
    ) -> AlertConfig:
        """Create an alert configuration."""
        config = AlertConfig(
            id=_gen_id(),
            suite_name=suite_name,
            team_id=team_id,
            pass_rate_threshold=pass_rate_threshold,
            cost_threshold=cost_threshold,
            latency_threshold_ms=latency_threshold_ms,
        )
        self._alerts[config.id] = config
        self._persist()
        return config

    def get_alert(self, alert_id: str) -> AlertConfig | None:
        return self._alerts.get(alert_id)

    def list_alerts(self, team_id: str | None = None) -> list[AlertConfig]:
        alerts = list(self._alerts.values())
        if team_id:
            alerts = [a for a in alerts if a.team_id == team_id]
        return alerts

    def check_alerts(self, run: SuiteRun) -> list[AlertEvent]:
        """Check if a run triggers any alerts."""
        events = []
        for config in self._alerts.values():
            if config.suite_name != run.suite_name:
                continue
            if config.team_id != run.team_id:
                continue
            if not config.enabled:
                continue

            # Check pass rate
            if run.overall_pass_rate < config.pass_rate_threshold:
                event = AlertEvent(
                    id=_gen_id(),
                    alert_config_id=config.id,
                    suite_name=run.suite_name,
                    team_id=run.team_id,
                    timestamp=_now(),
                    severity="critical",
                    message=(
                        f"Pass rate {run.overall_pass_rate:.0%} below "
                        f"threshold {config.pass_rate_threshold:.0%}"
                    ),
                    metric="pass_rate",
                    baseline_value=config.pass_rate_threshold,
                    current_value=run.overall_pass_rate,
                )
                self._alert_events[event.id] = event
                events.append(event)

            # Check cost
            if config.cost_threshold and run.total_cost > config.cost_threshold:
                event = AlertEvent(
                    id=_gen_id(),
                    alert_config_id=config.id,
                    suite_name=run.suite_name,
                    team_id=run.team_id,
                    timestamp=_now(),
                    severity="warning",
                    message=(
                        f"Cost ${run.total_cost:.4f} exceeds "
                        f"threshold ${config.cost_threshold:.4f}"
                    ),
                    metric="cost",
                    baseline_value=config.cost_threshold,
                    current_value=run.total_cost,
                )
                self._alert_events[event.id] = event
                events.append(event)

        self._persist()
        return events

    def list_alert_events(
        self,
        team_id: str | None = None,
        limit: int = 50,
    ) -> list[AlertEvent]:
        events = list(self._alert_events.values())
        if team_id:
            events = [e for e in events if e.team_id == team_id]
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]

    # --- Cost Reports ---

    def generate_cost_report(
        self,
        team_id: str,
        period_start: str,
        period_end: str,
    ) -> CostReport:
        """Generate a cost report for a team over a period."""
        runs = [
            r
            for r in self._runs.values()
            if r.team_id == team_id
            and period_start <= r.timestamp <= period_end
        ]

        cost_by_suite: dict[str, float] = {}
        cost_by_model: dict[str, float] = {}
        total_trials = 0

        for run in runs:
            cost_by_suite[run.suite_name] = (
                cost_by_suite.get(run.suite_name, 0) + run.total_cost
            )
            if run.model:
                cost_by_model[run.model] = (
                    cost_by_model.get(run.model, 0) + run.total_cost
                )
            total_trials += run.trials

        return CostReport(
            team_id=team_id,
            period_start=period_start,
            period_end=period_end,
            total_cost=sum(r.total_cost for r in runs),
            cost_by_suite=cost_by_suite,
            cost_by_model=cost_by_model,
            total_runs=len(runs),
            total_trials=total_trials,
        )

    # --- Persistence ---

    def _persist(self) -> None:
        """Save state to disk if data_dir is set."""
        if not self.data_dir:
            return

        self.data_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "teams": {k: _to_dict(v) for k, v in self._teams.items()},
            "runs": {k: _to_dict(v) for k, v in self._runs.items()},
            "alerts": {k: _to_dict(v) for k, v in self._alerts.items()},
            "alert_events": {
                k: _to_dict(v) for k, v in self._alert_events.items()
            },
        }
        with open(self.data_dir / "dashboard.json", "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load(self) -> None:
        """Load state from disk."""
        if not self.data_dir:
            return
        path = self.data_dir / "dashboard.json"
        if not path.exists():
            return

        with open(path) as f:
            state = json.load(f)

        for _id, data in state.get("teams", {}).items():
            self._teams[_id] = Team(**data)

        for _id, data in state.get("runs", {}).items():
            case_results = [
                CaseRunResult(**cr) for cr in data.pop("case_results", [])
            ]
            self._runs[_id] = SuiteRun(**data, case_results=case_results)

        for _id, data in state.get("alerts", {}).items():
            channels = data.pop("channels", [])
            self._alerts[_id] = AlertConfig(**data, channels=channels)

        for _id, data in state.get("alert_events", {}).items():
            self._alert_events[_id] = AlertEvent(**data)


def _gen_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass to dict recursively."""
    import dataclasses

    if dataclasses.is_dataclass(obj):
        result = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            result[f.name] = _to_dict(value)
        return result
    if isinstance(obj, list):
        return [_to_dict(item) for item in obj]  # type: ignore[return-value]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj
