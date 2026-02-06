"""Tests for eval packs discovery and loading."""

from agentrial.packs import (
    PackInfo,
    discover_packs,
    list_runtime_packs,
    load_runtime_pack,
    register_pack,
    _runtime_packs,
)
from agentrial.types import AgentInput, Suite, TestCase


def _make_suite(name: str = "test-pack-suite") -> Suite:
    return Suite(
        name=name,
        agent="dummy.agent",
        trials=5,
        threshold=0.8,
        cases=[
            TestCase(name="case-1", input=AgentInput(query="test")),
        ],
        tags=[],
    )


class TestDiscoverPacks:
    """Test entry point discovery."""

    def test_discover_returns_list(self):
        packs = discover_packs()
        assert isinstance(packs, list)

    def test_pack_info_structure(self):
        info = PackInfo(name="test", module="test_module", version="1.0")
        assert info.name == "test"
        assert info.version == "1.0"


class TestRuntimePacks:
    """Test programmatic pack registration."""

    def setup_method(self):
        """Clean up runtime packs before each test."""
        _runtime_packs.clear()

    def test_register_and_list(self):
        register_pack("my-pack", lambda: _make_suite())
        assert "my-pack" in list_runtime_packs()

    def test_load_runtime_pack_single_suite(self):
        register_pack("my-pack", lambda: _make_suite("single"))
        suites = load_runtime_pack("my-pack")
        assert len(suites) == 1
        assert suites[0].name == "single"

    def test_load_runtime_pack_list(self):
        register_pack("multi", lambda: [_make_suite("a"), _make_suite("b")])
        suites = load_runtime_pack("multi")
        assert len(suites) == 2

    def test_load_nonexistent_pack_raises(self):
        import pytest

        with pytest.raises(ValueError, match="not registered"):
            load_runtime_pack("nonexistent")

    def test_register_overwrites(self):
        register_pack("pack", lambda: _make_suite("v1"))
        register_pack("pack", lambda: _make_suite("v2"))
        suites = load_runtime_pack("pack")
        assert suites[0].name == "v2"
