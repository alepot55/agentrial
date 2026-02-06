"""Agentrial - Statistical evaluation framework for AI agents."""

from agentrial.evaluators.expect import expect
from agentrial.types import (
    AgentInput,
    AgentOutput,
    EvalResult,
    Suite,
    SuiteResult,
    TestCase,
    TrajectoryStep,
)

__version__ = "0.2.0"

__all__ = [
    "AgentInput",
    "AgentOutput",
    "TrajectoryStep",
    "EvalResult",
    "SuiteResult",
    "TestCase",
    "Suite",
    "expect",
    "__version__",
]
