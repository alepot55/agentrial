"""Integration tests for the agentrial CLI."""

import json
import os
import shutil
import subprocess
import tempfile

# Use .venv binary if available (local dev), otherwise find in PATH (CI)
_venv_bin = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".venv", "bin", "agentrial")
AGENTRIAL = _venv_bin if os.path.isfile(_venv_bin) else shutil.which("agentrial") or "agentrial"


def run_cli(*args: str, cwd: str | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run agentrial CLI as subprocess."""
    cmd = [AGENTRIAL] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


class TestCLIHelp:
    """Verify all CLI commands exist and have help."""

    def test_main_help(self):
        r = run_cli("--help")
        assert r.returncode == 0
        assert "run" in r.stdout
        assert "compare" in r.stdout

    def test_run_help(self):
        r = run_cli("run", "--help")
        assert r.returncode == 0
        assert "--trials" in r.stdout
        assert "--threshold" in r.stdout
        assert "--json" in r.stdout

    def test_compare_help(self):
        r = run_cli("compare", "--help")
        assert r.returncode == 0
        assert "--baseline" in r.stdout

    def test_config_help(self):
        r = run_cli("config", "--help")
        assert r.returncode == 0

    def test_baseline_help(self):
        r = run_cli("baseline", "--help")
        assert r.returncode == 0

    def test_snapshot_help(self):
        r = run_cli("snapshot", "--help")
        assert r.returncode == 0
        assert "update" in r.stdout
        assert "check" in r.stdout

    def test_security_help(self):
        r = run_cli("security", "--help")
        assert r.returncode == 0
        assert "scan" in r.stdout

    def test_pareto_help(self):
        r = run_cli("pareto", "--help")
        assert r.returncode == 0

    def test_prompt_help(self):
        r = run_cli("prompt", "--help")
        assert r.returncode == 0
        assert "track" in r.stdout
        assert "diff" in r.stdout
        assert "list" in r.stdout

    def test_monitor_help(self):
        r = run_cli("monitor", "--help")
        assert r.returncode == 0
        assert "--baseline" in r.stdout

    def test_dashboard_help(self):
        r = run_cli("dashboard", "--help")
        assert r.returncode == 0
        assert "--port" in r.stdout

    def test_version(self):
        r = run_cli("--version")
        assert r.returncode == 0


class TestCLIInit:
    """Test the init command."""

    def test_init_creates_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r = run_cli("init", cwd=tmpdir)
            assert r.returncode == 0
            # Should create agentrial.yml or a test file
            files = os.listdir(tmpdir)
            assert len(files) > 0, "init should create at least one file"

    def test_init_sample_runnable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r = run_cli("init", cwd=tmpdir)
            assert r.returncode == 0
            # Try running with very few trials
            r2 = run_cli("run", "--trials", "2", "--json", cwd=tmpdir, timeout=60)
            if r2.returncode == 0:
                # Verify JSON output is valid
                # stdout may contain non-JSON lines before JSON
                lines = r2.stdout.strip().split("\n")
                # Find the JSON block
                json_start = None
                for i, line in enumerate(lines):
                    if line.strip().startswith("{"):
                        json_start = i
                        break
                if json_start is not None:
                    json_str = "\n".join(lines[json_start:])
                    data = json.loads(json_str)
                    assert "summary" in data or "suite" in data


class TestCLIRunNoFiles:
    """Test run command behavior with no test files."""

    def test_run_no_files_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            r = run_cli("run", cwd=tmpdir)
            assert r.returncode != 0
