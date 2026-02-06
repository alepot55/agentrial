"""Local Benchmark Registry.

Provides a local registry for publishing and verifying agent benchmark
results. Results are stored as signed JSON files in a local directory,
enabling reproducible comparisons.

The registry stores:
- Agent name and version
- Suite name and configuration
- ARS score and breakdown
- Run metadata (date, trials, machine info)
- SHA-256 hash for integrity verification
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentrial.ars import ARSBreakdown

DEFAULT_REGISTRY_DIR = Path.home() / ".agentrial" / "registry"


@dataclass
class RegistryEntry:
    """A published benchmark result."""

    agent_name: str
    agent_version: str
    suite_name: str
    ars_score: float
    ars_breakdown: dict[str, Any]
    trials: int
    pass_rate: float
    total_cost: float
    total_duration_ms: float
    timestamp: str
    machine: str
    python_version: str
    hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class BenchmarkRegistry:
    """Local benchmark registry for publishing and querying results."""

    def __init__(self, registry_dir: str | Path | None = None) -> None:
        self.registry_dir = Path(registry_dir) if registry_dir else DEFAULT_REGISTRY_DIR
        self.registry_dir.mkdir(parents=True, exist_ok=True)

    def publish(
        self,
        agent_name: str,
        agent_version: str,
        suite_name: str,
        ars: ARSBreakdown,
        trials: int,
        pass_rate: float,
        total_cost: float,
        total_duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> RegistryEntry:
        """Publish a benchmark result to the registry.

        Args:
            agent_name: Name of the agent.
            agent_version: Version string of the agent.
            suite_name: Name of the test suite.
            ars: ARS breakdown from compute_ars().
            trials: Number of trials run.
            pass_rate: Overall pass rate.
            total_cost: Total cost.
            total_duration_ms: Total duration in ms.
            metadata: Optional extra metadata.

        Returns:
            The created RegistryEntry.
        """
        entry = RegistryEntry(
            agent_name=agent_name,
            agent_version=agent_version,
            suite_name=suite_name,
            ars_score=ars.score,
            ars_breakdown=asdict(ars),
            trials=trials,
            pass_rate=pass_rate,
            total_cost=total_cost,
            total_duration_ms=total_duration_ms,
            timestamp=datetime.now(UTC).isoformat(),
            machine=platform.node(),
            python_version=platform.python_version(),
            metadata=metadata or {},
        )

        # Compute integrity hash
        entry.hash = self._compute_hash(entry)

        # Save to file
        filename = f"{agent_name}_{agent_version}_{suite_name}.json"
        # Sanitize filename
        filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        path = self.registry_dir / filename
        with open(path, "w") as f:
            json.dump(asdict(entry), f, indent=2, default=str)

        return entry

    def list_entries(
        self,
        agent_name: str | None = None,
        suite_name: str | None = None,
    ) -> list[RegistryEntry]:
        """List registry entries with optional filters."""
        entries = []
        for path in sorted(self.registry_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                entry = RegistryEntry(**data)
                if agent_name and entry.agent_name != agent_name:
                    continue
                if suite_name and entry.suite_name != suite_name:
                    continue
                entries.append(entry)
            except (json.JSONDecodeError, TypeError):
                continue
        return entries

    def get_entry(
        self,
        agent_name: str,
        agent_version: str,
        suite_name: str,
    ) -> RegistryEntry | None:
        """Get a specific registry entry."""
        filename = f"{agent_name}_{agent_version}_{suite_name}.json"
        filename = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
        path = self.registry_dir / filename
        if not path.exists():
            return None
        with open(path) as f:
            data = json.load(f)
        return RegistryEntry(**data)

    def verify(self, entry: RegistryEntry) -> bool:
        """Verify the integrity of a registry entry.

        Returns True if the hash matches the entry data.
        """
        expected = self._compute_hash(entry)
        return expected == entry.hash

    @staticmethod
    def _compute_hash(entry: RegistryEntry) -> str:
        """Compute SHA-256 hash of the entry data (excluding hash field)."""
        data = {
            "agent_name": entry.agent_name,
            "agent_version": entry.agent_version,
            "suite_name": entry.suite_name,
            "ars_score": entry.ars_score,
            "trials": entry.trials,
            "pass_rate": entry.pass_rate,
            "total_cost": entry.total_cost,
            "total_duration_ms": entry.total_duration_ms,
            "timestamp": entry.timestamp,
        }
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
