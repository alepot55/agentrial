"""Tests for Prompt Version Control."""

from __future__ import annotations

import tempfile
from pathlib import Path

from agentrial.prompts import (
    PromptComparison,
    PromptDiff,
    PromptStore,
    PromptVersion,
    _compute_hash,
    compute_diff,
)


class TestComputeHash:
    """Tests for prompt hashing."""

    def test_deterministic(self) -> None:
        h1 = _compute_hash("hello world")
        h2 = _compute_hash("hello world")
        assert h1 == h2

    def test_different_inputs(self) -> None:
        h1 = _compute_hash("hello")
        h2 = _compute_hash("world")
        assert h1 != h2

    def test_short_hash(self) -> None:
        h = _compute_hash("test")
        assert len(h) == 12


class TestComputeDiff:
    """Tests for prompt diffing."""

    def test_identical_prompts(self) -> None:
        va = PromptVersion("v1", "Search for flights", "2025-01-01", "abc")
        vb = PromptVersion("v2", "Search for flights", "2025-01-02", "abc")
        diff = compute_diff(va, vb)

        assert not diff.changed
        assert diff.added_lines == 0
        assert diff.removed_lines == 0
        assert "No changes" in diff.summary

    def test_added_line(self) -> None:
        va = PromptVersion("v1", "Search for flights", "2025-01-01", "abc")
        vb = PromptVersion(
            "v2",
            "Search for flights\nAlways verify the price is in USD",
            "2025-01-02",
            "def",
        )
        diff = compute_diff(va, vb)

        assert diff.changed
        assert diff.added_lines >= 1
        assert "added" in diff.summary.lower()

    def test_removed_line(self) -> None:
        va = PromptVersion(
            "v1",
            "Search for flights\nInclude hotel recommendations",
            "2025-01-01",
            "abc",
        )
        vb = PromptVersion("v2", "Search for flights", "2025-01-02", "def")
        diff = compute_diff(va, vb)

        assert diff.changed
        assert diff.removed_lines >= 1

    def test_modified_line(self) -> None:
        va = PromptVersion("v1", "Search for cheap flights", "2025-01-01", "abc")
        vb = PromptVersion(
            "v2", "Search for the cheapest flights", "2025-01-02", "def"
        )
        diff = compute_diff(va, vb)

        assert diff.changed
        assert diff.text_diff != ""


class TestPromptStore:
    """Tests for PromptStore."""

    def test_track_auto_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            pv = store.track("Search for flights")

            assert pv.version == "v1"
            assert pv.prompt_text == "Search for flights"
            assert pv.hash == _compute_hash("Search for flights")

    def test_track_explicit_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            pv = store.track("prompt text", version="custom-v1")
            assert pv.version == "custom-v1"

    def test_track_with_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            pv = store.track(
                "prompt",
                tags={"model": "gpt-4o", "author": "alice"},
            )
            assert pv.tags["model"] == "gpt-4o"

    def test_get_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            store.track("version one")

            loaded = store.get_version("v1")
            assert loaded is not None
            assert loaded.prompt_text == "version one"

    def test_get_version_not_found(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            assert store.get_version("nonexistent") is None

    def test_list_versions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            store.track("first")
            store.track("second")
            store.track("third")

            versions = store.list_versions()
            assert len(versions) == 3
            assert "v1" in versions
            assert "v3" in versions

    def test_list_versions_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=Path(tmpdir) / "nonexistent")
            assert store.list_versions() == []

    def test_diff(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            store.track("Search for flights")
            store.track("Search for the cheapest flights")

            diff = store.diff("v1", "v2")
            assert diff.changed
            assert diff.version_a == "v1"
            assert diff.version_b == "v2"

    def test_diff_not_found(self) -> None:
        import pytest

        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            store.track("test")

            with pytest.raises(ValueError, match="not found"):
                store.diff("v1", "v99")

    def test_attach_eval_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            store.track("prompt text")

            store.attach_eval_results("v1", {
                "pass_rate": 0.85,
                "mean_cost": 0.12,
            })

            loaded = store.get_version("v1")
            assert loaded is not None
            assert loaded.eval_results is not None
            assert loaded.eval_results["pass_rate"] == 0.85

    def test_compare_with_eval_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            store.track("Search for flights")
            store.track("Search for the cheapest flights. Verify price in USD.")

            store.attach_eval_results("v1", {
                "pass_rate": 0.72,
                "mean_cost": 0.12,
            })
            store.attach_eval_results("v2", {
                "pass_rate": 0.88,
                "mean_cost": 0.14,
                "p_value": 0.04,
            })

            comp = store.compare("v1", "v2")
            assert comp.pass_rate_a == 0.72
            assert comp.pass_rate_b == 0.88
            assert comp.pass_rate_delta is not None
            assert comp.pass_rate_delta > 0
            assert comp.significant
            assert "improved" in comp.summary.lower()

    def test_compare_without_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            store.track("prompt a")
            store.track("prompt b")

            comp = store.compare("v1", "v2")
            assert comp.diff.changed
            assert comp.pass_rate_a is None
            assert comp.summary == ""


class TestPromptDiff:
    """Tests for PromptDiff dataclass."""

    def test_no_change(self) -> None:
        diff = PromptDiff(
            version_a="v1",
            version_b="v2",
            text_diff="",
            added_lines=0,
            removed_lines=0,
            changed=False,
        )
        assert not diff.changed

    def test_with_changes(self) -> None:
        diff = PromptDiff(
            version_a="v1",
            version_b="v2",
            text_diff="+new line",
            added_lines=1,
            removed_lines=0,
            changed=True,
            summary="1 line(s) added.",
        )
        assert diff.changed
        assert diff.added_lines == 1


class TestPromptComparison:
    """Tests for PromptComparison."""

    def test_significant_improvement(self) -> None:
        comp = PromptComparison(
            version_a="v1",
            version_b="v2",
            diff=PromptDiff("v1", "v2", "", 0, 0, True),
            pass_rate_a=0.72,
            pass_rate_b=0.88,
            pass_rate_delta=0.16,
            pass_rate_p=0.03,
            significant=True,
        )
        assert comp.significant

    def test_not_significant(self) -> None:
        comp = PromptComparison(
            version_a="v1",
            version_b="v2",
            diff=PromptDiff("v1", "v2", "", 0, 0, True),
            pass_rate_a=0.80,
            pass_rate_b=0.82,
            pass_rate_delta=0.02,
            pass_rate_p=0.45,
            significant=False,
        )
        assert not comp.significant


class TestCorruptPromptJSON:
    """Tests for corrupt JSON handling in PromptStore (M5)."""

    def test_corrupt_version_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            store.track("good prompt")  # v1

            # Corrupt the file
            corrupt_path = Path(tmpdir) / "v1.json"
            corrupt_path.write_text("{invalid json")

            result = store.get_version("v1")
            assert result is None

    def test_corrupt_file_skipped_in_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = PromptStore(base_dir=tmpdir)
            store.track("prompt a")  # v1
            store.track("prompt b")  # v2

            # Corrupt v1
            (Path(tmpdir) / "v1.json").write_text("not json!")

            versions = store.list_versions()
            assert "v1" not in versions
            assert "v2" in versions
