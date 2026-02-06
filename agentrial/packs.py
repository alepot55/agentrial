"""Eval Packs — discoverable evaluation pack system.

Eval packs are collections of test cases that can be published as
separate packages and discovered at runtime via Python entry points.

A pack is a Python package that provides a `agentrial.packs` entry point:

    # In the pack's pyproject.toml:
    [project.entry-points."agentrial.packs"]
    my_pack = "my_pack:get_suite"

The entry point function should return a Suite or list of Suite objects.
"""

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from typing import Any, Callable

from agentrial.types import Suite

ENTRY_POINT_GROUP = "agentrial.packs"


@dataclass
class PackInfo:
    """Information about a discovered eval pack."""

    name: str
    module: str
    version: str = ""
    description: str = ""
    suites: list[str] = field(default_factory=list)


def discover_packs() -> list[PackInfo]:
    """Discover installed eval packs via entry points.

    Returns:
        List of PackInfo for each discovered pack.
    """
    packs = []
    eps = importlib.metadata.entry_points()

    # Python 3.12+ returns a SelectableGroups or dict
    if hasattr(eps, "select"):
        group_eps = eps.select(group=ENTRY_POINT_GROUP)
    elif isinstance(eps, dict):
        group_eps = eps.get(ENTRY_POINT_GROUP, [])
    else:
        group_eps = [ep for ep in eps if ep.group == ENTRY_POINT_GROUP]

    for ep in group_eps:
        info = PackInfo(
            name=ep.name,
            module=ep.value,
        )

        # Try to get version from the dist that provides this entry point
        try:
            if hasattr(ep, "dist") and ep.dist is not None:
                info.version = ep.dist.version
                meta = ep.dist.metadata
                if meta:
                    info.description = meta.get("Summary", "")
        except Exception:
            pass

        # Try to load and get suite names
        try:
            loader = ep.load()
            result = loader() if callable(loader) else loader
            if isinstance(result, Suite):
                info.suites = [result.name]
            elif isinstance(result, list):
                info.suites = [s.name for s in result if isinstance(s, Suite)]
        except Exception:
            pass

        packs.append(info)

    return packs


def load_pack(name: str) -> list[Suite]:
    """Load suites from a named eval pack.

    Args:
        name: The pack name (entry point name).

    Returns:
        List of Suite objects from the pack.

    Raises:
        ValueError: If the pack is not found.
    """
    eps = importlib.metadata.entry_points()

    if hasattr(eps, "select"):
        group_eps = list(eps.select(group=ENTRY_POINT_GROUP, name=name))
    elif isinstance(eps, dict):
        group_eps = [ep for ep in eps.get(ENTRY_POINT_GROUP, []) if ep.name == name]
    else:
        group_eps = [ep for ep in eps if ep.group == ENTRY_POINT_GROUP and ep.name == name]

    if not group_eps:
        raise ValueError(f"Eval pack '{name}' not found. Run 'agentrial packs list' to see available packs.")

    ep = group_eps[0]
    loader = ep.load()
    result = loader() if callable(loader) else loader

    if isinstance(result, Suite):
        return [result]
    if isinstance(result, list):
        return [s for s in result if isinstance(s, Suite)]

    raise ValueError(
        f"Pack '{name}' returned {type(result).__name__}, expected Suite or list[Suite]"
    )


def register_pack(
    name: str,
    loader: Callable[[], Suite | list[Suite]],
) -> None:
    """Register a pack programmatically (for runtime-only packs).

    This is an alternative to entry points for packs that are
    defined dynamically rather than installed as packages.

    Args:
        name: Pack name.
        loader: Callable that returns Suite or list of Suite.
    """
    _runtime_packs[name] = loader


def list_runtime_packs() -> list[str]:
    """List names of programmatically registered packs."""
    return list(_runtime_packs.keys())


def load_runtime_pack(name: str) -> list[Suite]:
    """Load a runtime-registered pack."""
    if name not in _runtime_packs:
        raise ValueError(f"Runtime pack '{name}' not registered.")
    result = _runtime_packs[name]()
    if isinstance(result, Suite):
        return [result]
    if isinstance(result, list):
        return [s for s in result if isinstance(s, Suite)]
    raise ValueError(f"Runtime pack '{name}' returned invalid type: {type(result).__name__}")


# Internal storage for runtime packs
_runtime_packs: dict[str, Callable[[], Suite | list[Suite]]] = {}
