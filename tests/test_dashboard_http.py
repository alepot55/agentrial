"""Tests for Cloud Dashboard HTTP endpoints using FastAPI TestClient."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from agentrial.dashboard.app import create_app  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the dashboard app."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


class TestTeamsEndpoints:
    """Tests for /api/teams endpoints."""

    def test_create_team(self, client: TestClient) -> None:
        response = client.post(
            "/api/teams",
            json={"name": "test-team", "members": ["alice", "bob"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test-team"
        assert "id" in data
        assert data["id"] != ""

    def test_list_teams(self, client: TestClient) -> None:
        client.post("/api/teams", json={"name": "team-a"})
        client.post("/api/teams", json={"name": "team-b"})

        response = client.get("/api/teams")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2
        names = [t["name"] for t in data]
        assert "team-a" in names
        assert "team-b" in names

    def test_get_team(self, client: TestClient) -> None:
        create_resp = client.post("/api/teams", json={"name": "get-me"})
        team_id = create_resp.json()["id"]

        response = client.get(f"/api/teams/{team_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "get-me"

    def test_get_nonexistent_team(self, client: TestClient) -> None:
        response = client.get("/api/teams/nonexistent-id")
        assert response.status_code == 404


class TestRunsEndpoints:
    """Tests for /api/runs endpoints."""

    def test_record_and_list_runs(self, client: TestClient) -> None:
        # Create a team first
        team_resp = client.post("/api/teams", json={"name": "run-team"})
        team_id = team_resp.json()["id"]

        # Record a run
        run_resp = client.post(
            "/api/runs",
            json={
                "suite_name": "test-suite",
                "team_id": team_id,
                "overall_pass_rate": 0.85,
                "total_cost": 1.20,
                "total_duration_ms": 5000,
                "passed": True,
                "trials": 10,
                "model": "gpt-4o",
            },
        )
        assert run_resp.status_code == 200
        run_data = run_resp.json()
        assert "id" in run_data
        assert "alerts_triggered" in run_data

        # List runs
        list_resp = client.get("/api/runs", params={"team_id": team_id})
        assert list_resp.status_code == 200
        runs = list_resp.json()
        assert len(runs) >= 1
        assert runs[0]["suite_name"] == "test-suite"
        assert runs[0]["overall_pass_rate"] == 0.85

    def test_get_run(self, client: TestClient) -> None:
        team_resp = client.post("/api/teams", json={"name": "get-run-team"})
        team_id = team_resp.json()["id"]

        run_resp = client.post(
            "/api/runs",
            json={
                "suite_name": "s",
                "team_id": team_id,
                "overall_pass_rate": 0.9,
                "total_cost": 0.5,
                "total_duration_ms": 1000,
                "passed": True,
                "case_results": [
                    {
                        "case_name": "case-1",
                        "pass_rate": 0.9,
                        "mean_cost": 0.05,
                        "trials_passed": 9,
                        "trials_total": 10,
                    }
                ],
            },
        )
        run_id = run_resp.json()["id"]

        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["overall_pass_rate"] == 0.9
        assert len(data["case_results"]) == 1
        assert data["case_results"][0]["case_name"] == "case-1"

    def test_get_nonexistent_run(self, client: TestClient) -> None:
        response = client.get("/api/runs/nonexistent-id")
        assert response.status_code == 404


class TestTrendsEndpoint:
    """Tests for GET /api/trends."""

    def test_get_trends(self, client: TestClient) -> None:
        team_resp = client.post("/api/teams", json={"name": "trend-team"})
        team_id = team_resp.json()["id"]

        for rate in [0.80, 0.82, 0.85]:
            client.post(
                "/api/runs",
                json={
                    "suite_name": "trend-suite",
                    "team_id": team_id,
                    "overall_pass_rate": rate,
                    "total_cost": 0.1,
                    "total_duration_ms": 100,
                    "passed": True,
                },
            )

        response = client.get(f"/api/trends/{team_id}/trend-suite")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        rates = [d["pass_rate"] for d in data]
        assert rates == [0.80, 0.82, 0.85]


class TestAlertsEndpoints:
    """Tests for /api/alerts endpoints."""

    def test_create_and_list_alerts(self, client: TestClient) -> None:
        team_resp = client.post("/api/teams", json={"name": "alert-team"})
        team_id = team_resp.json()["id"]

        alert_resp = client.post(
            "/api/alerts",
            json={
                "suite_name": "alert-suite",
                "team_id": team_id,
                "pass_rate_threshold": 0.85,
            },
        )
        assert alert_resp.status_code == 200
        assert "id" in alert_resp.json()

        list_resp = client.get("/api/alerts", params={"team_id": team_id})
        assert list_resp.status_code == 200
        alerts = list_resp.json()
        assert len(alerts) >= 1
        assert alerts[0]["pass_rate_threshold"] == 0.85

    def test_alert_events(self, client: TestClient) -> None:
        team_resp = client.post("/api/teams", json={"name": "event-team"})
        team_id = team_resp.json()["id"]

        # Create alert
        client.post(
            "/api/alerts",
            json={
                "suite_name": "event-suite",
                "team_id": team_id,
                "pass_rate_threshold": 0.85,
            },
        )

        # Record a failing run to trigger alert
        client.post(
            "/api/runs",
            json={
                "suite_name": "event-suite",
                "team_id": team_id,
                "overall_pass_rate": 0.5,
                "total_cost": 1.0,
                "total_duration_ms": 5000,
                "passed": False,
            },
        )

        response = client.get("/api/alert-events", params={"team_id": team_id})
        assert response.status_code == 200
        events = response.json()
        assert len(events) >= 1
        assert events[0]["severity"] == "critical"


class TestCostReportEndpoint:
    """Tests for GET /api/cost-report."""

    def test_cost_report(self, client: TestClient) -> None:
        team_resp = client.post("/api/teams", json={"name": "cost-team"})
        team_id = team_resp.json()["id"]

        client.post(
            "/api/runs",
            json={
                "suite_name": "cost-suite",
                "team_id": team_id,
                "overall_pass_rate": 0.9,
                "total_cost": 1.50,
                "total_duration_ms": 1000,
                "passed": True,
                "trials": 10,
                "model": "gpt-4o",
            },
        )

        response = client.get(f"/api/cost-report/{team_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["team_id"] == team_id
        assert data["total_cost"] == 1.50
        assert data["total_runs"] == 1
