"""Tests for terminal and JSON reporters."""

import json

from agentrial.reporters.json_report import (
    compare_reports,
    export_json,
    load_json_report,
    save_json_report,
)
from agentrial.reporters.terminal import (
    format_ci,
    format_cost,
    format_latency,
    format_percentage,
    get_pass_rate_style,
    print_results,
)
from agentrial.types import (
    AgentInput,
    AgentMetadata,
    AgentOutput,
    ConfidenceInterval,
    EvalResult,
    Suite,
    SuiteResult,
    TestCase,
    TrialResult,
)


def _make_suite_result(pass_rate: float = 0.9, num_cases: int = 2) -> SuiteResult:
    """Create a minimal SuiteResult for testing."""
    suite = Suite(
        name="test-suite",
        agent="test.agent",
        trials=10,
        threshold=0.85,
        cases=[],
        tags=[],
    )

    results = []
    for i in range(num_cases):
        tc = TestCase(
            name=f"test-case-{i}",
            input=AgentInput(query=f"query {i}"),
        )
        trials = []
        for j in range(10):
            passed = j < int(pass_rate * 10)
            trials.append(
                TrialResult(
                    trial_index=j,
                    test_case=tc,
                    agent_output=AgentOutput(
                        output=f"output {j}",
                        steps=[],
                        metadata=AgentMetadata(
                            total_tokens=100,
                            cost=0.01,
                            duration_ms=500.0,
                        ),
                        success=passed,
                    ),
                    passed=passed,
                    duration_ms=500.0,
                    cost=0.01,
                    tokens=100,
                )
            )
        results.append(
            EvalResult(
                test_case=tc,
                trials=trials,
                pass_rate=pass_rate,
                pass_rate_ci=ConfidenceInterval(lower=0.7, upper=0.98),
                mean_cost=0.01,
                cost_ci=ConfidenceInterval(lower=0.008, upper=0.012),
                mean_latency_ms=500.0,
                latency_ci=ConfidenceInterval(lower=400.0, upper=600.0),
                mean_tokens=100.0,
            )
        )

    return SuiteResult(
        suite=suite,
        results=results,
        overall_pass_rate=pass_rate,
        overall_pass_rate_ci=ConfidenceInterval(lower=0.7, upper=0.98),
        total_cost=0.2,
        total_duration_ms=10000.0,
        passed=pass_rate >= 0.85,
    )


class TestTerminalFormatters:
    """Test terminal formatting functions."""

    def test_format_percentage(self):
        assert "90" in format_percentage(0.9)

    def test_format_ci(self):
        result = format_ci(0.7, 0.95)
        assert "70" in result
        assert "95" in result

    def test_format_cost_small(self):
        result = format_cost(0.002)
        assert "$" in result

    def test_format_cost_large(self):
        result = format_cost(1.5)
        assert "$" in result

    def test_format_latency_ms(self):
        result = format_latency(450.0)
        assert "ms" in result or "s" in result

    def test_format_latency_seconds(self):
        result = format_latency(2500.0)
        assert "s" in result

    def test_pass_rate_style_green(self):
        style = get_pass_rate_style(0.95, 0.85)
        assert "green" in style

    def test_pass_rate_style_red(self):
        style = get_pass_rate_style(0.3, 0.85)
        assert "red" in style


class TestTerminalPrint:
    """Test that print functions don't crash."""

    def test_print_results_no_crash(self):
        sr = _make_suite_result()
        # Should not raise
        print_results(sr)

    def test_print_results_verbose(self):
        sr = _make_suite_result()
        print_results(sr, verbose=True)

    def test_print_results_failing_suite(self):
        sr = _make_suite_result(pass_rate=0.4)
        print_results(sr)


class TestJsonExport:
    """Test JSON export/import."""

    def test_export_valid_json(self):
        sr = _make_suite_result()
        data = export_json(sr)
        assert isinstance(data, dict)
        assert "version" in data
        assert "timestamp" in data
        assert "summary" in data
        assert "results" in data

    def test_export_summary_fields(self):
        sr = _make_suite_result(pass_rate=0.9)
        data = export_json(sr)
        summary = data["summary"]
        assert "passed" in summary
        assert "overall_pass_rate" in summary
        assert summary["passed"] is True

    def test_save_and_load_roundtrip(self, tmp_path):
        sr = _make_suite_result()
        path = tmp_path / "results.json"
        save_json_report(sr, path)
        loaded = load_json_report(path)
        assert loaded["summary"]["overall_pass_rate"] == 0.9

    def test_exported_json_is_serializable(self):
        sr = _make_suite_result()
        data = export_json(sr)
        # Should not raise
        json_str = json.dumps(data)
        assert len(json_str) > 0


class TestJsonCompare:
    """Test JSON report comparison."""

    def test_compare_same_reports(self):
        sr = _make_suite_result()
        data = export_json(sr)
        comparison = compare_reports(data, data)
        assert isinstance(comparison, dict)

    def test_compare_different_pass_rates(self):
        sr_good = _make_suite_result(pass_rate=0.9)
        sr_bad = _make_suite_result(pass_rate=0.5)
        comparison = compare_reports(export_json(sr_bad), export_json(sr_good))
        assert isinstance(comparison, dict)
