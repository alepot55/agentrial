"""Tests for OpenTelemetry span capture."""

from agentrial.runner.otel import OTelTrajectoryCapture, TrajectorySpanExporter


class TestTrajectorySpanExporter:
    """Tests for the span exporter."""

    def test_init(self):
        exporter = TrajectorySpanExporter()
        assert exporter.get_steps() == []

    def test_clear(self):
        exporter = TrajectorySpanExporter()
        exporter.clear()
        assert exporter.get_steps() == []

    def test_shutdown(self):
        exporter = TrajectorySpanExporter()
        exporter.shutdown()

    def test_force_flush(self):
        exporter = TrajectorySpanExporter()
        assert exporter.force_flush() is True


class TestOTelTrajectoryCapture:
    """Tests for the context manager."""

    def test_context_manager(self):
        with OTelTrajectoryCapture() as capture:
            steps = capture.get_steps()
            assert isinstance(steps, list)
            assert len(steps) == 0

    def test_clear_in_context(self):
        with OTelTrajectoryCapture() as capture:
            capture.clear()
            assert capture.get_steps() == []

    def test_nested_context_restores_provider(self):
        """Ensure provider is restored after exit."""
        with OTelTrajectoryCapture():
            pass
        # Should not crash
        with OTelTrajectoryCapture():
            pass
