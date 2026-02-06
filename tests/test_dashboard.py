"""Tests for Cloud Dashboard store and models."""

from __future__ import annotations

import tempfile

from agentrial.dashboard.models import (
    CaseRunResult,
    ComparisonView,
    SuiteRun,
    TraceStep,
    TraceView,
)
from agentrial.dashboard.store import DashboardStore


class TestTeamManagement:
    """Tests for team CRUD."""

    def test_create_team(self) -> None:
        store = DashboardStore()
        team = store.create_team("engineering", members=["alice", "bob"])

        assert team.name == "engineering"
        assert "alice" in team.members
        assert team.id != ""

    def test_get_team(self) -> None:
        store = DashboardStore()
        team = store.create_team("test-team")

        loaded = store.get_team(team.id)
        assert loaded is not None
        assert loaded.name == "test-team"

    def test_get_nonexistent_team(self) -> None:
        store = DashboardStore()
        assert store.get_team("nonexistent") is None

    def test_list_teams(self) -> None:
        store = DashboardStore()
        store.create_team("team-a")
        store.create_team("team-b")

        teams = store.list_teams()
        assert len(teams) == 2


class TestSuiteRuns:
    """Tests for suite run recording."""

    def test_record_run(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")

        run = store.record_run(
            suite_name="flight-search",
            team_id=team.id,
            overall_pass_rate=0.85,
            total_cost=1.20,
            total_duration_ms=5000,
            passed=True,
            trials=10,
            model="gpt-4o",
        )

        assert run.suite_name == "flight-search"
        assert run.overall_pass_rate == 0.85
        assert run.id != ""

    def test_get_run(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")
        run = store.record_run(
            suite_name="test",
            team_id=team.id,
            overall_pass_rate=0.9,
            total_cost=0.5,
            total_duration_ms=1000,
            passed=True,
        )

        loaded = store.get_run(run.id)
        assert loaded is not None
        assert loaded.overall_pass_rate == 0.9

    def test_list_runs_filtered(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")

        store.record_run("suite-a", team.id, 0.9, 0.1, 100, True)
        store.record_run("suite-b", team.id, 0.8, 0.2, 200, True)
        store.record_run("suite-a", team.id, 0.85, 0.15, 150, True)

        all_runs = store.list_runs(team_id=team.id)
        assert len(all_runs) == 3

        suite_a = store.list_runs(suite_name="suite-a")
        assert len(suite_a) == 2

    def test_list_runs_with_limit(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")
        for i in range(10):
            store.record_run(f"suite-{i}", team.id, 0.9, 0.1, 100, True)

        limited = store.list_runs(limit=5)
        assert len(limited) == 5

    def test_record_with_case_results(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")

        cases = [
            CaseRunResult(
                case_name="search",
                pass_rate=0.9,
                mean_cost=0.05,
                trials_passed=9,
                trials_total=10,
            ),
            CaseRunResult(
                case_name="booking",
                pass_rate=0.7,
                mean_cost=0.08,
                trials_passed=7,
                trials_total=10,
            ),
        ]

        run = store.record_run(
            suite_name="flight",
            team_id=team.id,
            overall_pass_rate=0.8,
            total_cost=1.3,
            total_duration_ms=3000,
            passed=True,
            case_results=cases,
        )

        assert len(run.case_results) == 2
        assert run.case_results[0].case_name == "search"


class TestTrends:
    """Tests for trend data."""

    def test_get_trend(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")

        for rate in [0.8, 0.82, 0.85, 0.83, 0.88]:
            store.record_run("suite", team.id, rate, 0.1, 100, True)

        trend = store.get_trend("suite", team.id)
        assert len(trend) == 5
        # Should be in chronological order
        rates = [t["pass_rate"] for t in trend]
        assert rates == [0.8, 0.82, 0.85, 0.83, 0.88]


class TestComparisons:
    """Tests for run comparison."""

    def test_compare_runs(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")

        run_a = store.record_run(
            "suite",
            team.id,
            0.8,
            1.0,
            5000,
            True,
            case_results=[
                CaseRunResult("case1", pass_rate=0.9),
                CaseRunResult("case2", pass_rate=0.7),
            ],
        )
        run_b = store.record_run(
            "suite",
            team.id,
            0.9,
            0.8,
            4000,
            True,
            case_results=[
                CaseRunResult("case1", pass_rate=0.95),
                CaseRunResult("case2", pass_rate=0.85),
            ],
        )

        comp = store.compare_runs(run_a.id, run_b.id)
        assert comp is not None
        assert comp.pass_rate_delta > 0
        assert comp.cost_delta < 0
        assert len(comp.improved_cases) > 0

    def test_compare_not_found(self) -> None:
        store = DashboardStore()
        assert store.compare_runs("nonexistent", "also_nonexistent") is None


class TestAlerts:
    """Tests for alert system."""

    def test_create_alert(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")

        config = store.create_alert(
            suite_name="flight",
            team_id=team.id,
            pass_rate_threshold=0.85,
        )
        assert config.id != ""
        assert config.pass_rate_threshold == 0.85

    def test_alert_triggered_on_regression(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")

        store.create_alert("flight", team.id, pass_rate_threshold=0.85)

        run = store.record_run("flight", team.id, 0.6, 1.0, 5000, False)
        events = store.check_alerts(run)

        assert len(events) == 1
        assert events[0].severity == "critical"
        assert "pass rate" in events[0].message.lower()

    def test_alert_not_triggered_above_threshold(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")

        store.create_alert("flight", team.id, pass_rate_threshold=0.8)

        run = store.record_run("flight", team.id, 0.9, 1.0, 5000, True)
        events = store.check_alerts(run)

        assert len(events) == 0

    def test_cost_alert(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")

        store.create_alert(
            "flight",
            team.id,
            pass_rate_threshold=0.0,
            cost_threshold=0.5,
        )

        run = store.record_run("flight", team.id, 0.9, 2.0, 5000, True)
        events = store.check_alerts(run)

        cost_events = [e for e in events if e.metric == "cost"]
        assert len(cost_events) == 1

    def test_list_alert_events(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")
        store.create_alert("suite", team.id, pass_rate_threshold=0.9)

        run = store.record_run("suite", team.id, 0.5, 1.0, 5000, False)
        store.check_alerts(run)

        events = store.list_alert_events(team_id=team.id)
        assert len(events) >= 1


class TestCostReports:
    """Tests for cost reporting."""

    def test_generate_report(self) -> None:
        store = DashboardStore()
        team = store.create_team("test")

        store.record_run(
            "suite-a", team.id, 0.9, 0.50, 100, True,
            trials=10, model="gpt-4o",
        )
        store.record_run(
            "suite-a", team.id, 0.85, 0.30, 100, True,
            trials=10, model="claude-sonnet",
        )
        store.record_run(
            "suite-b", team.id, 0.8, 0.20, 100, True,
            trials=5, model="gpt-4o",
        )

        report = store.generate_cost_report(
            team.id,
            "2000-01-01",
            "2099-12-31",
        )

        assert report.total_cost == 1.0
        assert report.total_runs == 3
        assert report.total_trials == 25
        assert "suite-a" in report.cost_by_suite
        assert "gpt-4o" in report.cost_by_model

    def test_empty_report(self) -> None:
        store = DashboardStore()
        report = store.generate_cost_report("none", "2000-01-01", "2099-12-31")
        assert report.total_cost == 0.0
        assert report.total_runs == 0


class TestPersistence:
    """Tests for file persistence."""

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate store
            store = DashboardStore(data_dir=tmpdir)
            team = store.create_team("persistent-team")
            store.record_run("suite", team.id, 0.9, 0.5, 1000, True)
            store.create_alert("suite", team.id, pass_rate_threshold=0.8)

            # Load fresh store from same directory
            store2 = DashboardStore(data_dir=tmpdir)
            teams = store2.list_teams()
            assert len(teams) == 1
            assert teams[0].name == "persistent-team"

            runs = store2.list_runs()
            assert len(runs) == 1

            alerts = store2.list_alerts()
            assert len(alerts) == 1


class TestModels:
    """Tests for data model constructors."""

    def test_trace_step(self) -> None:
        step = TraceStep(
            step_index=0,
            step_type="tool_call",
            name="search",
            parameters={"q": "flights"},
            tokens=100,
        )
        assert step.name == "search"
        assert step.tokens == 100

    def test_trace_view(self) -> None:
        view = TraceView(
            run_id="abc",
            case_name="test",
            trial_index=0,
            passed=True,
            steps=[
                TraceStep(0, "tool_call", "search"),
                TraceStep(1, "llm_call", "response"),
            ],
        )
        assert len(view.steps) == 2

    def test_comparison_view(self) -> None:
        run_a = SuiteRun("a", "suite", "t1", "2025-01-01", overall_pass_rate=0.8)
        run_b = SuiteRun("b", "suite", "t1", "2025-01-02", overall_pass_rate=0.9)
        comp = ComparisonView(
            run_a=run_a,
            run_b=run_b,
            pass_rate_delta=0.1,
            improved_cases=["case1"],
        )
        assert comp.pass_rate_delta == 0.1
