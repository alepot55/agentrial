"""FastAPI application for the Cloud Dashboard.

Provides REST API endpoints for:
- Team management
- Suite run recording and listing
- Run comparison
- Alert configuration and events
- Cost reporting
- Trend data for charts

Usage:
    # Development server
    uvicorn agentrial.dashboard.app:create_app --factory --reload

    # Or via CLI
    agentrial dashboard --port 8080
"""

from __future__ import annotations

from typing import Any

from agentrial.dashboard.models import CaseRunResult
from agentrial.dashboard.store import DashboardStore


def create_app(
    data_dir: str | None = None,
) -> Any:
    """Create the FastAPI application.

    Args:
        data_dir: Optional directory for persistent storage.

    Returns:
        FastAPI application instance.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError as err:
        raise ImportError(
            "FastAPI is not installed. Install it with: "
            "pip install 'agentrial[dashboard]'  "
            "(requires fastapi and uvicorn)"
        ) from err

    app = FastAPI(
        title="Agentrial Dashboard",
        description="Cloud dashboard for agent evaluation results",
        version="0.1.0",
    )

    # CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    store = DashboardStore(data_dir=data_dir)

    # --- Health ---

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # --- Teams ---

    @app.post("/api/teams")
    def create_team(body: dict[str, Any]) -> dict[str, Any]:
        team = store.create_team(
            name=body["name"],
            members=body.get("members", []),
        )
        return {"id": team.id, "name": team.name}

    @app.get("/api/teams")
    def list_teams() -> list[dict[str, Any]]:
        return [
            {"id": t.id, "name": t.name, "members": t.members}
            for t in store.list_teams()
        ]

    @app.get("/api/teams/{team_id}")
    def get_team(team_id: str) -> dict[str, Any]:
        team = store.get_team(team_id)
        if not team:
            raise HTTPException(404, f"Team {team_id} not found")
        return {"id": team.id, "name": team.name, "members": team.members}

    # --- Runs ---

    @app.post("/api/runs")
    def record_run(body: dict[str, Any]) -> dict[str, Any]:
        case_results = [
            CaseRunResult(**cr) for cr in body.get("case_results", [])
        ]
        run = store.record_run(
            suite_name=body["suite_name"],
            team_id=body["team_id"],
            overall_pass_rate=body["overall_pass_rate"],
            total_cost=body.get("total_cost", 0.0),
            total_duration_ms=body.get("total_duration_ms", 0.0),
            passed=body.get("passed", False),
            trials=body.get("trials", 0),
            model=body.get("model", ""),
            prompt_version=body.get("prompt_version", ""),
            case_results=case_results,
            metadata=body.get("metadata", {}),
        )

        # Check alerts
        alert_events = store.check_alerts(run)

        return {
            "id": run.id,
            "alerts_triggered": len(alert_events),
        }

    @app.get("/api/runs")
    def list_runs(
        team_id: str | None = None,
        suite_name: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        runs = store.list_runs(
            team_id=team_id,
            suite_name=suite_name,
            limit=limit,
        )
        return [
            {
                "id": r.id,
                "suite_name": r.suite_name,
                "timestamp": r.timestamp,
                "overall_pass_rate": r.overall_pass_rate,
                "total_cost": r.total_cost,
                "passed": r.passed,
                "model": r.model,
            }
            for r in runs
        ]

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = store.get_run(run_id)
        if not run:
            raise HTTPException(404, f"Run {run_id} not found")
        return {
            "id": run.id,
            "suite_name": run.suite_name,
            "team_id": run.team_id,
            "timestamp": run.timestamp,
            "model": run.model,
            "overall_pass_rate": run.overall_pass_rate,
            "total_cost": run.total_cost,
            "total_duration_ms": run.total_duration_ms,
            "passed": run.passed,
            "trials": run.trials,
            "case_results": [
                {
                    "case_name": cr.case_name,
                    "pass_rate": cr.pass_rate,
                    "mean_cost": cr.mean_cost,
                    "mean_latency_ms": cr.mean_latency_ms,
                }
                for cr in run.case_results
            ],
        }

    # --- Comparisons ---

    @app.get("/api/compare/{run_id_a}/{run_id_b}")
    def compare_runs(run_id_a: str, run_id_b: str) -> dict[str, Any]:
        comparison = store.compare_runs(run_id_a, run_id_b)
        if not comparison:
            raise HTTPException(404, "One or both runs not found")
        return {
            "pass_rate_delta": comparison.pass_rate_delta,
            "cost_delta": comparison.cost_delta,
            "duration_delta": comparison.duration_delta,
            "improved_cases": comparison.improved_cases,
            "regressed_cases": comparison.regressed_cases,
            "unchanged_cases": comparison.unchanged_cases,
        }

    # --- Trends ---

    @app.get("/api/trends/{team_id}/{suite_name}")
    def get_trend(
        team_id: str,
        suite_name: str,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        return store.get_trend(suite_name, team_id, limit)

    # --- Alerts ---

    @app.post("/api/alerts")
    def create_alert(body: dict[str, Any]) -> dict[str, Any]:
        config = store.create_alert(
            suite_name=body["suite_name"],
            team_id=body["team_id"],
            pass_rate_threshold=body.get("pass_rate_threshold", 0.8),
            cost_threshold=body.get("cost_threshold"),
            latency_threshold_ms=body.get("latency_threshold_ms"),
        )
        return {"id": config.id}

    @app.get("/api/alerts")
    def list_alerts(team_id: str | None = None) -> list[dict[str, Any]]:
        alerts = store.list_alerts(team_id)
        return [
            {
                "id": a.id,
                "suite_name": a.suite_name,
                "enabled": a.enabled,
                "pass_rate_threshold": a.pass_rate_threshold,
            }
            for a in alerts
        ]

    @app.get("/api/alert-events")
    def list_alert_events(
        team_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        events = store.list_alert_events(team_id, limit)
        return [
            {
                "id": e.id,
                "suite_name": e.suite_name,
                "timestamp": e.timestamp,
                "severity": e.severity,
                "message": e.message,
                "metric": e.metric,
                "acknowledged": e.acknowledged,
            }
            for e in events
        ]

    # --- Cost Reports ---

    @app.get("/api/cost-report/{team_id}")
    def cost_report(
        team_id: str,
        period_start: str = "",
        period_end: str = "",
    ) -> dict[str, Any]:
        if not period_start:
            period_start = "2000-01-01"
        if not period_end:
            period_end = "2099-12-31"

        report = store.generate_cost_report(team_id, period_start, period_end)
        return {
            "team_id": report.team_id,
            "period_start": report.period_start,
            "period_end": report.period_end,
            "total_cost": report.total_cost,
            "cost_by_suite": report.cost_by_suite,
            "cost_by_model": report.cost_by_model,
            "total_runs": report.total_runs,
            "total_trials": report.total_trials,
        }

    return app
