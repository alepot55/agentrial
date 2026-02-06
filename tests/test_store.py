"""Tests for the dashboard store."""

from agentrial.dashboard.store import DashboardStore


class TestDashboardStoreTeams:
    """Test team operations."""

    def test_create_team(self, tmp_path):
        store = DashboardStore(data_dir=tmp_path)
        team = store.create_team("test-team", members=["alice", "bob"])
        assert team.name == "test-team"
        assert team.id is not None
        assert "alice" in team.members

    def test_get_team(self, tmp_path):
        store = DashboardStore(data_dir=tmp_path)
        team = store.create_team("test-team")
        fetched = store.get_team(team.id)
        assert fetched is not None
        assert fetched.name == "test-team"

    def test_get_nonexistent_team(self, tmp_path):
        store = DashboardStore(data_dir=tmp_path)
        assert store.get_team("nonexistent") is None

    def test_list_teams(self, tmp_path):
        store = DashboardStore(data_dir=tmp_path)
        store.create_team("team-a")
        store.create_team("team-b")
        teams = store.list_teams()
        assert len(teams) == 2


class TestDashboardStoreRuns:
    """Test suite run operations."""

    def test_record_and_get_run(self, tmp_path):
        store = DashboardStore(data_dir=tmp_path)
        team = store.create_team("test-team")
        run = store.record_run(
            suite_name="my-suite",
            team_id=team.id,
            overall_pass_rate=0.9,
            total_cost=1.5,
            total_duration_ms=5000.0,
            passed=True,
            trials=100,
        )
        assert run.suite_name == "my-suite"
        fetched = store.get_run(run.id)
        assert fetched is not None
        assert fetched.overall_pass_rate == 0.9

    def test_list_runs(self, tmp_path):
        store = DashboardStore(data_dir=tmp_path)
        team = store.create_team("test-team")
        store.record_run("s1", team.id, 0.9, 1.0, 1000, True)
        store.record_run("s2", team.id, 0.8, 2.0, 2000, False)
        runs = store.list_runs()
        assert len(runs) == 2

    def test_list_runs_filter_by_suite(self, tmp_path):
        store = DashboardStore(data_dir=tmp_path)
        team = store.create_team("test-team")
        store.record_run("alpha", team.id, 0.9, 1.0, 1000, True)
        store.record_run("beta", team.id, 0.8, 2.0, 2000, False)
        runs = store.list_runs(suite_name="alpha")
        assert len(runs) == 1
        assert runs[0].suite_name == "alpha"


class TestDashboardStoreAlerts:
    """Test alert operations."""

    def test_create_alert(self, tmp_path):
        store = DashboardStore(data_dir=tmp_path)
        team = store.create_team("test-team")
        alert = store.create_alert("my-suite", team.id, pass_rate_threshold=0.8)
        assert alert.suite_name == "my-suite"
        assert alert.pass_rate_threshold == 0.8

    def test_list_alerts(self, tmp_path):
        store = DashboardStore(data_dir=tmp_path)
        team = store.create_team("test-team")
        store.create_alert("s1", team.id)
        store.create_alert("s2", team.id)
        alerts = store.list_alerts()
        assert len(alerts) == 2

    def test_check_alerts_triggers(self, tmp_path):
        store = DashboardStore(data_dir=tmp_path)
        team = store.create_team("test-team")
        store.create_alert("my-suite", team.id, pass_rate_threshold=0.9)
        # Record a run below threshold
        run = store.record_run("my-suite", team.id, 0.5, 1.0, 1000, False)
        events = store.check_alerts(run)
        assert len(events) >= 1


class TestDashboardStorePersistence:
    """Test data persistence."""

    def test_persist_and_reload(self, tmp_path):
        store1 = DashboardStore(data_dir=tmp_path)
        team = store1.create_team("persist-team")
        store1.record_run("my-suite", team.id, 0.85, 1.0, 1000, True)

        # Create new store instance, should load persisted data
        store2 = DashboardStore(data_dir=tmp_path)
        teams = store2.list_teams()
        assert len(teams) == 1
        assert teams[0].name == "persist-team"
        runs = store2.list_runs()
        assert len(runs) == 1
