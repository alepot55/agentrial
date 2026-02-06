"""Configuration loading for Agentrial."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agentrial.types import (
    AgentInput,
    ExpectedOutput,
    StepExpectation,
    Suite,
    TestCase,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILENAME = "agentrial.yml"


@dataclass
class Config:
    """Global configuration for Agentrial.

    Attributes:
        trials: Default number of trials per test case.
        threshold: Default minimum pass rate.
        output_format: Output format (terminal, json, both).
        json_output_path: Path for JSON output.
        verbose: Enable verbose output.
        parallel: Number of parallel workers (1 = sequential).
    """

    trials: int = 10
    threshold: float = 0.85
    output_format: str = "terminal"
    json_output_path: str | None = None
    verbose: bool = False
    parallel: int = 1


def load_config(config_path: Path | None = None) -> Config:
    """Load configuration from file.

    Args:
        config_path: Path to config file. If None, looks for agenteval.yml
            in current directory.

    Returns:
        Loaded configuration.
    """
    if config_path is None:
        config_path = Path.cwd() / DEFAULT_CONFIG_FILENAME

    if not config_path.exists():
        logger.debug("No config file found at %s, using defaults", config_path)
        return Config()

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return Config(
        trials=data.get("trials", 10),
        threshold=data.get("threshold", 0.85),
        output_format=data.get("output_format", "terminal"),
        json_output_path=data.get("json_output_path"),
        verbose=data.get("verbose", False),
        parallel=data.get("parallel", 1),
    )


def _parse_expected_output(data: dict[str, Any] | None) -> ExpectedOutput | None:
    """Parse expected output from YAML data."""
    if not data:
        return None

    return ExpectedOutput(
        exact_match=data.get("exact_match"),
        contains=data.get("output_contains") or data.get("contains"),
        contains_any=data.get("output_contains_any") or data.get("contains_any"),
        regex=data.get("regex"),
        tool_calls=data.get("tool_calls"),
    )


def _parse_step_expectations(steps: list[dict[str, Any]] | None) -> list[StepExpectation]:
    """Parse step expectations from YAML data."""
    if not steps:
        return []

    expectations = []
    for i, step in enumerate(steps):
        expectations.append(
            StepExpectation(
                step_index=step.get("step_index", i),
                name=step.get("name"),
                expected_tool=step.get("expected_tool"),
                params_contain=step.get("params_contain"),
                output_contains=step.get("output_contains"),
            )
        )
    return expectations


def _parse_test_case(data: dict[str, Any]) -> TestCase:
    """Parse a test case from YAML data."""
    input_data = data.get("input", {})
    if isinstance(input_data, str):
        agent_input = AgentInput(query=input_data)
    else:
        agent_input = AgentInput(
            query=input_data.get("query", ""),
            context=input_data.get("context", {}),
        )

    return TestCase(
        name=data["name"],
        input=agent_input,
        expected=_parse_expected_output(data.get("expected")),
        step_expectations=_parse_step_expectations(data.get("steps")),
        max_cost=data.get("max_cost"),
        max_latency_ms=data.get("max_latency_ms"),
        tags=data.get("tags", []),
    )


def load_suite_from_yaml(yaml_path: Path) -> Suite:
    """Load a test suite from a YAML file.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Loaded test suite.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the YAML is invalid.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Test file not found: {yaml_path}")

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not data:
        raise ValueError(f"Empty or invalid YAML file: {yaml_path}")

    cases = [_parse_test_case(case) for case in data.get("cases", [])]

    return Suite(
        name=data.get("suite", yaml_path.stem),
        agent=data.get("agent", ""),
        trials=data.get("trials", 10),
        threshold=data.get("threshold", 0.85),
        cases=cases,
        tags=data.get("tags", []),
    )


def load_suite_from_python(python_path: Path) -> Suite:
    """Load a test suite from a Python file.

    The Python file should define a `suite` variable of type Suite.

    Args:
        python_path: Path to the Python file.

    Returns:
        Loaded test suite.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If no suite is found in the file.
    """
    import importlib.util

    if not python_path.exists():
        raise FileNotFoundError(f"Test file not found: {python_path}")

    spec = importlib.util.spec_from_file_location("test_module", python_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load Python file: {python_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Look for a Suite instance in the module
    suite = getattr(module, "suite", None)
    if suite is None:
        # Look for any Suite instance
        for name in dir(module):
            obj = getattr(module, name)
            if isinstance(obj, Suite):
                suite = obj
                break

    if suite is None:
        raise ValueError(f"No Suite found in {python_path}")

    return suite


def load_suite(path: Path) -> Suite:
    """Load a test suite from a file (YAML or Python).

    Args:
        path: Path to the test file.

    Returns:
        Loaded test suite.
    """
    if path.suffix in (".yml", ".yaml"):
        return load_suite_from_yaml(path)
    elif path.suffix == ".py":
        return load_suite_from_python(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")


def discover_test_files(directory: Path, pattern: str = "test_*.yml") -> list[Path]:
    """Discover test files in a directory.

    Looks for test files with the following patterns:
    - test_*.yml / test_*.yaml (standard test files)
    - test_*.py (Python test files)
    - agentrial.yml / agentrial.yaml (project-level test suites)

    Args:
        directory: Directory to search.
        pattern: Glob pattern for test files.

    Returns:
        List of test file paths.
    """
    files = list(directory.glob(pattern))
    # Also look for YAML variants
    files.extend(directory.glob("test_*.yaml"))
    # Also look for Python test files
    files.extend(directory.glob("test_*.py"))
    # Also look for agentrial.yml/yaml as valid test suite files
    for name in ("agentrial.yml", "agentrial.yaml"):
        agentrial_file = directory / name
        if agentrial_file.exists():
            files.append(agentrial_file)
    return sorted(set(files))  # Remove duplicates and sort
