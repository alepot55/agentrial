"""Prompt Version Control.

Tracks, diffs, and evaluates prompt versions. Prompts are the
"source code" of agents — this module versions them like git
versions code, with statistical comparison between versions.

Usage:
    from agentrial.prompts import PromptStore
    store = PromptStore()
    store.track("Search for the cheapest flight", tags={"model": "gpt-4o"})
    diff = store.diff("v1", "v2")
    print(diff.text_diff)
"""

from __future__ import annotations

import difflib
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROMPTS_DIR = ".agentrial/prompts"


@dataclass
class PromptVersion:
    """A versioned prompt snapshot."""

    version: str
    prompt_text: str
    created_at: str
    hash: str
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    eval_results: dict[str, Any] | None = None


@dataclass
class PromptDiff:
    """Diff between two prompt versions."""

    version_a: str
    version_b: str
    text_diff: str
    added_lines: int
    removed_lines: int
    changed: bool
    summary: str = ""


@dataclass
class PromptComparison:
    """Statistical comparison between two prompt versions."""

    version_a: str
    version_b: str
    diff: PromptDiff
    pass_rate_a: float | None = None
    pass_rate_b: float | None = None
    pass_rate_delta: float | None = None
    pass_rate_p: float | None = None
    cost_a: float | None = None
    cost_b: float | None = None
    cost_delta: float | None = None
    significant: bool = False
    summary: str = ""


def _compute_hash(text: str) -> str:
    """Compute a short hash of prompt text."""
    return hashlib.sha256(text.encode()).hexdigest()[:12]


class PromptStore:
    """Store and manage prompt versions.

    Persists prompt versions as JSON files in .agentrial/prompts/.
    """

    def __init__(self, base_dir: str | Path | None = None) -> None:
        """Initialize the store.

        Args:
            base_dir: Directory for prompt storage. Defaults to .agentrial/prompts/.
        """
        self.base_dir = Path(base_dir) if base_dir else Path(PROMPTS_DIR)

    def track(
        self,
        prompt_text: str,
        version: str | None = None,
        tags: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PromptVersion:
        """Track a new prompt version.

        Args:
            prompt_text: The prompt text to version.
            version: Explicit version string. Auto-generated if not provided.
            tags: Optional tags (e.g., model, author).
            metadata: Optional metadata dict.

        Returns:
            The created PromptVersion.
        """
        prompt_hash = _compute_hash(prompt_text)

        if version is None:
            # Auto-generate version from existing count
            existing = self.list_versions()
            version = f"v{len(existing) + 1}"

        pv = PromptVersion(
            version=version,
            prompt_text=prompt_text,
            created_at=datetime.now(UTC).isoformat(),
            hash=prompt_hash,
            tags=tags or {},
            metadata=metadata or {},
        )

        self._save_version(pv)
        return pv

    def get_version(self, version: str) -> PromptVersion | None:
        """Get a specific prompt version.

        Args:
            version: Version string (e.g., "v1").

        Returns:
            PromptVersion or None if not found.
        """
        path = self.base_dir / f"{version}.json"
        if not path.exists():
            return None

        try:
            with open(path) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            import logging

            logging.getLogger(__name__).warning(
                "Corrupt JSON in prompt version file '%s' — skipping", path
            )
            return None

        return PromptVersion(**data)

    def list_versions(self) -> list[str]:
        """List all tracked versions in order.

        Returns:
            List of version strings (skips corrupt files).
        """
        if not self.base_dir.exists():
            return []

        versions = []
        for path in sorted(self.base_dir.glob("*.json")):
            # Verify the file is valid JSON before listing
            try:
                with open(path) as f:
                    json.load(f)
                versions.append(path.stem)
            except json.JSONDecodeError:
                import logging

                logging.getLogger(__name__).warning(
                    "Corrupt JSON in prompt version file '%s' — skipping", path
                )
        return versions

    def diff(self, version_a: str, version_b: str) -> PromptDiff:
        """Compute diff between two prompt versions.

        Args:
            version_a: First version string.
            version_b: Second version string.

        Returns:
            PromptDiff with textual diff and statistics.

        Raises:
            ValueError: If either version not found.
        """
        va = self.get_version(version_a)
        vb = self.get_version(version_b)

        if va is None:
            raise ValueError(f"Version '{version_a}' not found")
        if vb is None:
            raise ValueError(f"Version '{version_b}' not found")

        return compute_diff(va, vb)

    def attach_eval_results(
        self,
        version: str,
        results: dict[str, Any],
    ) -> None:
        """Attach evaluation results to a prompt version.

        Args:
            version: Version to attach results to.
            results: Evaluation results dict (pass_rate, cost, etc.).

        Raises:
            ValueError: If version not found.
        """
        pv = self.get_version(version)
        if pv is None:
            raise ValueError(f"Version '{version}' not found")

        pv.eval_results = results
        self._save_version(pv)

    def compare(
        self,
        version_a: str,
        version_b: str,
    ) -> PromptComparison:
        """Compare two versions statistically using attached eval results.

        Args:
            version_a: First version.
            version_b: Second version.

        Returns:
            PromptComparison with diff and statistical analysis.
        """
        va = self.get_version(version_a)
        vb = self.get_version(version_b)

        if va is None:
            raise ValueError(f"Version '{version_a}' not found")
        if vb is None:
            raise ValueError(f"Version '{version_b}' not found")

        d = compute_diff(va, vb)

        comparison = PromptComparison(
            version_a=version_a,
            version_b=version_b,
            diff=d,
        )

        # Extract eval results if available
        if va.eval_results and vb.eval_results:
            ra = va.eval_results
            rb = vb.eval_results

            comparison.pass_rate_a = ra.get("pass_rate")
            comparison.pass_rate_b = rb.get("pass_rate")
            comparison.cost_a = ra.get("mean_cost")
            comparison.cost_b = rb.get("mean_cost")

            if comparison.pass_rate_a is not None and comparison.pass_rate_b is not None:
                comparison.pass_rate_delta = (
                    comparison.pass_rate_b - comparison.pass_rate_a
                )

            if comparison.cost_a is not None and comparison.cost_b is not None:
                comparison.cost_delta = comparison.cost_b - comparison.cost_a

            # Statistical significance if p-value available
            comparison.pass_rate_p = ra.get("p_value") or rb.get("p_value")
            if comparison.pass_rate_p is not None:
                comparison.significant = comparison.pass_rate_p < 0.05

            # Generate summary
            comparison.summary = _generate_comparison_summary(comparison)

        return comparison

    def _save_version(self, pv: PromptVersion) -> None:
        """Save a version to disk."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        path = self.base_dir / f"{pv.version}.json"

        data = {
            "version": pv.version,
            "prompt_text": pv.prompt_text,
            "created_at": pv.created_at,
            "hash": pv.hash,
            "tags": pv.tags,
            "metadata": pv.metadata,
            "eval_results": pv.eval_results,
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def compute_diff(va: PromptVersion, vb: PromptVersion) -> PromptDiff:
    """Compute a textual diff between two prompt versions.

    Args:
        va: First version.
        vb: Second version.

    Returns:
        PromptDiff with unified diff and statistics.
    """
    lines_a = va.prompt_text.splitlines(keepends=True)
    lines_b = vb.prompt_text.splitlines(keepends=True)

    diff_lines = list(
        difflib.unified_diff(
            lines_a,
            lines_b,
            fromfile=f"prompt {va.version}",
            tofile=f"prompt {vb.version}",
        )
    )

    text_diff = "".join(diff_lines)
    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

    changed = va.hash != vb.hash

    # Generate summary
    if not changed:
        summary = "No changes between versions."
    elif added > 0 and removed > 0:
        summary = f"{added} line(s) added, {removed} line(s) removed."
    elif added > 0:
        summary = f"{added} line(s) added."
    else:
        summary = f"{removed} line(s) removed."

    return PromptDiff(
        version_a=va.version,
        version_b=vb.version,
        text_diff=text_diff,
        added_lines=added,
        removed_lines=removed,
        changed=changed,
        summary=summary,
    )


def _generate_comparison_summary(comp: PromptComparison) -> str:
    """Generate a human-readable comparison summary."""
    parts = []

    if comp.pass_rate_delta is not None:
        direction = "improved" if comp.pass_rate_delta > 0 else "regressed"
        sig = " (significant)" if comp.significant else " (not significant)"
        parts.append(
            f"Pass rate {direction} by {abs(comp.pass_rate_delta):.0%}"
            f" ({comp.pass_rate_a:.0%} -> {comp.pass_rate_b:.0%}){sig}"
        )

    if comp.cost_delta is not None:
        direction = "increased" if comp.cost_delta > 0 else "decreased"
        parts.append(
            f"Cost {direction} by ${abs(comp.cost_delta):.4f}"
            f" (${comp.cost_a:.4f} -> ${comp.cost_b:.4f})"
        )

    return " | ".join(parts) if parts else "No evaluation data to compare."


def print_prompt_diff(
    diff: PromptDiff,
    console: Any = None,
) -> None:
    """Print a prompt diff to terminal.

    Args:
        diff: The prompt diff to display.
        console: Rich console (creates one if not provided).
    """
    from rich.console import Console
    from rich.syntax import Syntax

    if console is None:
        console = Console()

    console.print(f"\n[bold]Prompt diff: {diff.version_a} -> {diff.version_b}[/bold]")
    console.print(f"[dim]{diff.summary}[/dim]")

    if diff.text_diff:
        syntax = Syntax(diff.text_diff, "diff", theme="monokai")
        console.print(syntax)
    else:
        console.print("[dim]No textual changes.[/dim]")


def print_prompt_comparison(
    comp: PromptComparison,
    console: Any = None,
) -> None:
    """Print a prompt comparison to terminal.

    Args:
        comp: The comparison to display.
        console: Rich console (creates one if not provided).
    """
    from rich.console import Console
    from rich.table import Table

    if console is None:
        console = Console()

    print_prompt_diff(comp.diff, console)

    if comp.pass_rate_a is not None:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column(comp.version_a, justify="right")
        table.add_column(comp.version_b, justify="right")
        table.add_column("Delta", justify="right")
        table.add_column("Sig.", justify="center")

        # Pass rate row
        delta_str = ""
        if comp.pass_rate_delta is not None:
            color = "green" if comp.pass_rate_delta > 0 else "red"
            delta_str = f"[{color}]{comp.pass_rate_delta:+.0%}[/{color}]"

        sig_str = ""
        if comp.pass_rate_p is not None:
            sig_str = f"p={comp.pass_rate_p:.3f}"

        table.add_row(
            "Pass Rate",
            f"{comp.pass_rate_a:.0%}" if comp.pass_rate_a is not None else "-",
            f"{comp.pass_rate_b:.0%}" if comp.pass_rate_b is not None else "-",
            delta_str,
            sig_str,
        )

        # Cost row
        if comp.cost_a is not None:
            cost_delta = ""
            if comp.cost_delta is not None:
                color = "red" if comp.cost_delta > 0 else "green"
                cost_delta = f"[{color}]${comp.cost_delta:+.4f}[/{color}]"

            table.add_row(
                "Cost",
                f"${comp.cost_a:.4f}",
                f"${comp.cost_b:.4f}" if comp.cost_b is not None else "-",
                cost_delta,
                "",
            )

        console.print(table)

    if comp.summary:
        console.print(f"\n[bold]Summary:[/bold] {comp.summary}")
