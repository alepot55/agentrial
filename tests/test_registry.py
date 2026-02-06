"""Tests for the benchmark registry."""

from agentrial.ars import ARSBreakdown
from agentrial.registry import BenchmarkRegistry, RegistryEntry


def _make_ars(score: float = 75.0) -> ARSBreakdown:
    return ARSBreakdown(
        score=score,
        accuracy=90.0,
        consistency=80.0,
        cost_efficiency=70.0,
        latency=60.0,
        trajectory_quality=100.0,
        recovery=50.0,
    )


class TestBenchmarkRegistry:
    """Test publish, list, get, verify."""

    def test_publish(self, tmp_path):
        registry = BenchmarkRegistry(registry_dir=tmp_path)
        entry = registry.publish(
            agent_name="my-agent",
            agent_version="1.0.0",
            suite_name="my-suite",
            ars=_make_ars(),
            trials=100,
            pass_rate=0.9,
            total_cost=1.5,
            total_duration_ms=5000.0,
        )
        assert entry.agent_name == "my-agent"
        assert entry.hash != ""

    def test_list_entries(self, tmp_path):
        registry = BenchmarkRegistry(registry_dir=tmp_path)
        registry.publish("a", "1.0", "s1", _make_ars(), 10, 0.9, 1.0, 1000.0)
        registry.publish("b", "1.0", "s1", _make_ars(), 10, 0.8, 2.0, 2000.0)
        entries = registry.list_entries()
        assert len(entries) == 2

    def test_list_filter_by_agent(self, tmp_path):
        registry = BenchmarkRegistry(registry_dir=tmp_path)
        registry.publish("a", "1.0", "s1", _make_ars(), 10, 0.9, 1.0, 1000.0)
        registry.publish("b", "1.0", "s1", _make_ars(), 10, 0.8, 2.0, 2000.0)
        entries = registry.list_entries(agent_name="a")
        assert len(entries) == 1
        assert entries[0].agent_name == "a"

    def test_get_entry(self, tmp_path):
        registry = BenchmarkRegistry(registry_dir=tmp_path)
        registry.publish("a", "1.0", "s1", _make_ars(), 10, 0.9, 1.0, 1000.0)
        entry = registry.get_entry("a", "1.0", "s1")
        assert entry is not None
        assert entry.agent_name == "a"

    def test_get_nonexistent_entry(self, tmp_path):
        registry = BenchmarkRegistry(registry_dir=tmp_path)
        entry = registry.get_entry("nope", "1.0", "s1")
        assert entry is None

    def test_verify_valid(self, tmp_path):
        registry = BenchmarkRegistry(registry_dir=tmp_path)
        entry = registry.publish("a", "1.0", "s1", _make_ars(), 10, 0.9, 1.0, 1000.0)
        assert registry.verify(entry) is True

    def test_verify_tampered(self, tmp_path):
        registry = BenchmarkRegistry(registry_dir=tmp_path)
        entry = registry.publish("a", "1.0", "s1", _make_ars(), 10, 0.9, 1.0, 1000.0)
        entry.pass_rate = 0.99  # Tamper
        assert registry.verify(entry) is False

    def test_publish_creates_file(self, tmp_path):
        registry = BenchmarkRegistry(registry_dir=tmp_path)
        registry.publish("a", "1.0", "s1", _make_ars(), 10, 0.9, 1.0, 1000.0)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 1


class TestRegistryEntry:
    """Test the RegistryEntry dataclass."""

    def test_fields(self):
        entry = RegistryEntry(
            agent_name="test",
            agent_version="1.0",
            suite_name="suite",
            ars_score=75.0,
            ars_breakdown={},
            trials=10,
            pass_rate=0.9,
            total_cost=1.0,
            total_duration_ms=5000.0,
            timestamp="2024-01-01T00:00:00",
            machine="test-machine",
            python_version="3.12.0",
        )
        assert entry.agent_name == "test"
        assert entry.hash == ""
        assert entry.metadata == {}
