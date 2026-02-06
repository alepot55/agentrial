"""Tests for framework adapters.

Since the actual frameworks (CrewAI, AutoGen, Pydantic AI, etc.) are not
installed in the test environment, these tests verify:
1. Import errors produce clear messages
2. Adapter classes correctly process mock framework outputs
3. TrajectoryRecorder integration works properly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentrial.runner.adapters.base import FunctionAdapter, wrap_function
from agentrial.types import AgentInput, AgentOutput, StepType

# --- FunctionAdapter tests ---


class TestFunctionAdapter:
    """Tests for the base FunctionAdapter."""

    def test_wrap_agent_output(self) -> None:
        def my_agent(inp: AgentInput) -> AgentOutput:
            return AgentOutput(output=f"Answer: {inp.query}")

        adapter = FunctionAdapter(my_agent)
        result = adapter(AgentInput(query="hello"))
        assert result.output == "Answer: hello"
        assert result.success

    def test_wrap_dict_result(self) -> None:
        def my_agent(inp: AgentInput) -> dict:
            return {
                "output": "result",
                "metadata": {"tokens": 100, "cost": 0.01},
            }

        adapter = FunctionAdapter(my_agent)
        result = adapter(AgentInput(query="test"))
        assert result.output == "result"
        assert result.metadata.total_tokens == 100

    def test_wrap_string_result(self) -> None:
        def my_agent(inp: AgentInput) -> str:
            return "just a string"

        adapter = FunctionAdapter(my_agent)
        result = adapter(AgentInput(query="test"))
        assert result.output == "just a string"

    def test_wrap_function_shortcut(self) -> None:
        def my_agent(inp: AgentInput) -> str:
            return "wrapped"

        fn = wrap_function(my_agent, name="test_agent")
        result = fn(AgentInput(query="test"))
        assert result.output == "wrapped"

    def test_get_agent_callable(self) -> None:
        def my_agent(inp: AgentInput) -> str:
            return "ok"

        adapter = FunctionAdapter(my_agent)
        assert adapter.get_agent_callable() is adapter

    def test_none_return_is_failure(self) -> None:
        """Agent returning None should produce success=False (M1)."""

        def none_agent(inp: AgentInput):
            return None

        adapter = FunctionAdapter(none_agent)
        result = adapter(AgentInput(query="test"))
        assert not result.success
        assert result.output == ""
        assert "None" in result.error


# --- smolagents naming tests ---


class TestSmolAgentsNaming:
    """Tests for wrap_smolagents_agent naming (M2)."""

    def test_new_name_exists(self) -> None:
        from agentrial.runner.adapters.smolagents import wrap_smolagents_agent

        assert callable(wrap_smolagents_agent)

    def test_alias_still_works(self) -> None:
        from agentrial.runner.adapters.smolagents import (
            wrap_smolagent,
            wrap_smolagents_agent,
        )

        assert wrap_smolagent is wrap_smolagents_agent

    def test_lazy_import_new_name(self) -> None:
        from agentrial.runner.adapters import wrap_smolagents_agent

        assert callable(wrap_smolagents_agent)


# --- CrewAI adapter tests ---


class TestCrewAIAdapter:
    """Tests for CrewAI adapter."""

    def test_import_error_message(self) -> None:
        from agentrial.runner.adapters.crewai import _import_crewai

        with patch.dict("sys.modules", {"crewai": None}):
            with pytest.raises(ImportError, match="CrewAI is not installed"):
                _import_crewai()

    def test_adapter_processes_crew_result(self) -> None:
        from agentrial.runner.adapters.crewai import CrewAIAdapter

        # Mock a crew with task outputs
        @dataclass
        class MockTaskOutput:
            agent: str = "researcher"
            description: str = "Search for data"

            def __str__(self) -> str:
                return "Found 10 results"

        @dataclass
        class MockTokenUsage:
            prompt_tokens: int = 100
            completion_tokens: int = 50
            total_cost: float = 0.01

        @dataclass
        class MockCrewResult:
            tasks_output: list = field(default_factory=list)
            token_usage: MockTokenUsage = field(default_factory=MockTokenUsage)

            def __str__(self) -> str:
                return "Final crew output"

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = MockCrewResult(
            tasks_output=[MockTaskOutput()],
            token_usage=MockTokenUsage(),
        )

        with patch("agentrial.runner.adapters.crewai._import_crewai"):
            adapter = CrewAIAdapter(mock_crew)
            result = adapter(AgentInput(query="research AI"))

        assert result.output == "Final crew output"
        assert result.success
        assert len(result.steps) > 0
        assert result.steps[0].name == "task_researcher"

    def test_adapter_handles_error(self) -> None:
        from agentrial.runner.adapters.crewai import CrewAIAdapter

        mock_crew = MagicMock()
        mock_crew.kickoff.side_effect = RuntimeError("Crew failed")

        with patch("agentrial.runner.adapters.crewai._import_crewai"):
            adapter = CrewAIAdapter(mock_crew)
            result = adapter(AgentInput(query="test"))

        assert not result.success
        assert "Crew failed" in result.error


# --- AutoGen adapter tests ---


class TestAutoGenAdapter:
    """Tests for AutoGen adapter."""

    def test_import_error_message(self) -> None:
        from agentrial.runner.adapters.autogen import _import_autogen

        with patch.dict(
            "sys.modules", {"autogen_agentchat": None, "autogen": None}
        ):
            with pytest.raises(ImportError, match="AutoGen is not installed"):
                _import_autogen()

    def test_legacy_adapter_processes_chat(self) -> None:
        from agentrial.runner.adapters.autogen import AutoGenAdapter

        @dataclass
        class MockChatResult:
            chat_history: list = field(default_factory=list)
            summary: str = "Chat summary"
            cost: dict = field(default_factory=dict)

        mock_agent = MagicMock()

        # Mock legacy autogen module
        mock_autogen = MagicMock()
        mock_user_proxy = MagicMock()
        mock_user_proxy.initiate_chat.return_value = MockChatResult(
            chat_history=[
                {"role": "user", "content": "Hello", "name": "user"},
                {
                    "role": "assistant",
                    "content": "Hi there!",
                    "name": "assistant",
                },
            ],
            summary="Hi there!",
        )
        mock_autogen.UserProxyAgent.return_value = mock_user_proxy

        with patch.dict("sys.modules", {"autogen_agentchat": None}):
            with patch.dict("sys.modules", {"autogen": mock_autogen}):
                adapter = AutoGenAdapter(mock_agent)
                result = adapter(AgentInput(query="Hello"))

        assert result.success
        assert result.output == "Hi there!"

    def test_adapter_handles_error(self) -> None:
        from agentrial.runner.adapters.autogen import AutoGenAdapter

        mock_agent = MagicMock()

        # Mock legacy autogen that raises on initiate_chat
        mock_autogen = MagicMock()
        mock_user_proxy = MagicMock()
        mock_user_proxy.initiate_chat.side_effect = RuntimeError("Connection failed")
        mock_autogen.UserProxyAgent.return_value = mock_user_proxy

        with patch.dict("sys.modules", {"autogen_agentchat": None}):
            with patch.dict("sys.modules", {"autogen": mock_autogen}):
                adapter = AutoGenAdapter(mock_agent)
                result = adapter(AgentInput(query="test"))

        assert not result.success
        assert "Connection failed" in result.error


# --- Pydantic AI adapter tests ---


class TestPydanticAIAdapter:
    """Tests for Pydantic AI adapter."""

    def test_import_error_message(self) -> None:
        from agentrial.runner.adapters.pydantic_ai import _import_pydantic_ai

        with patch.dict("sys.modules", {"pydantic_ai": None}):
            with pytest.raises(ImportError, match="Pydantic AI is not installed"):
                _import_pydantic_ai()

    def test_adapter_processes_result(self) -> None:
        from agentrial.runner.adapters.pydantic_ai import PydanticAIAdapter

        @dataclass
        class MockUsage:
            request_tokens: int = 100
            response_tokens: int = 50

        @dataclass
        class MockTextPart:
            part_kind: str = "text"
            content: str = "Analysis complete"

        @dataclass
        class MockToolCallPart:
            part_kind: str = "tool-call"
            tool_name: str = "search"
            args: dict = field(default_factory=dict)
            tool_call_id: str = "tc_1"

        @dataclass
        class MockResponseMsg:
            kind: str = "response"
            parts: list = field(default_factory=list)

        @dataclass
        class MockResult:
            data: str = "Final answer"

            def all_messages(self):
                return [
                    MockResponseMsg(
                        parts=[
                            MockToolCallPart(args={"q": "test"}),
                            MockTextPart(),
                        ]
                    )
                ]

            def usage(self):
                return MockUsage()

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = MockResult()

        adapter = PydanticAIAdapter(mock_agent)
        result = adapter(AgentInput(query="analyze this"))

        assert result.output == "Final answer"
        assert result.success
        assert len(result.steps) == 2  # tool_call + text
        assert result.steps[0].step_type == StepType.TOOL_CALL
        assert result.steps[0].name == "search"

    def test_adapter_handles_error(self) -> None:
        from agentrial.runner.adapters.pydantic_ai import PydanticAIAdapter

        mock_agent = MagicMock()
        mock_agent.run_sync.side_effect = RuntimeError("Model unavailable")

        adapter = PydanticAIAdapter(mock_agent)
        result = adapter(AgentInput(query="test"))

        assert not result.success
        assert "Model unavailable" in result.error


# --- OpenAI Agents SDK adapter tests ---


class TestOpenAIAgentsAdapter:
    """Tests for OpenAI Agents SDK adapter."""

    def test_import_error_message(self) -> None:
        from agentrial.runner.adapters.openai_agents import _import_openai_agents

        with patch.dict("sys.modules", {"agents": None}):
            with pytest.raises(
                ImportError, match="OpenAI Agents SDK is not installed"
            ):
                _import_openai_agents()

    def test_adapter_processes_result(self) -> None:
        from agentrial.runner.adapters.openai_agents import OpenAIAgentsAdapter

        @dataclass
        class MockAgent:
            name: str = "assistant"

        @dataclass
        class MockToolCallItem:
            type: str = "tool_call_item"
            agent: Any = field(default_factory=lambda: MockAgent())
            raw_item: Any = None

            def __post_init__(self):
                if self.raw_item is None:
                    self.raw_item = MagicMock(
                        name="search", arguments='{"q": "test"}', call_id="c1"
                    )

        @dataclass
        class MockTextContent:
            text: str = "Here is the answer"

        @dataclass
        class MockRawMessage:
            content: list = field(
                default_factory=lambda: [MockTextContent()]
            )

        @dataclass
        class MockMessageItem:
            type: str = "message_output_item"
            agent: Any = field(default_factory=lambda: MockAgent())
            raw_item: Any = field(default_factory=lambda: MockRawMessage())

        @dataclass
        class MockRunResult:
            final_output: str = "Done"
            new_items: list = field(default_factory=list)
            raw_responses: list = field(default_factory=list)

        # Create mock SDK module
        mock_sdk = MagicMock()
        mock_sdk.Runner.run_sync.return_value = MockRunResult(
            new_items=[MockToolCallItem(), MockMessageItem()],
        )

        with patch(
            "agentrial.runner.adapters.openai_agents._import_openai_agents",
            return_value=mock_sdk,
        ):
            adapter = OpenAIAgentsAdapter(MagicMock())
            result = adapter(AgentInput(query="test"))

        assert result.output == "Done"
        assert result.success
        assert len(result.steps) >= 2

    def test_adapter_handles_error(self) -> None:
        from agentrial.runner.adapters.openai_agents import OpenAIAgentsAdapter

        mock_sdk = MagicMock()
        mock_sdk.Runner.run_sync.side_effect = RuntimeError("API error")

        with patch(
            "agentrial.runner.adapters.openai_agents._import_openai_agents",
            return_value=mock_sdk,
        ):
            adapter = OpenAIAgentsAdapter(MagicMock())
            result = adapter(AgentInput(query="test"))

        assert not result.success
        assert "API error" in result.error


# --- smolagents adapter tests ---


class TestSmolAgentsAdapter:
    """Tests for smolagents (Hugging Face) adapter."""

    def test_import_error_message(self) -> None:
        from agentrial.runner.adapters.smolagents import _import_smolagents

        with patch.dict("sys.modules", {"smolagents": None}):
            with pytest.raises(ImportError, match="smolagents is not installed"):
                _import_smolagents()

    def test_adapter_processes_dict_logs(self) -> None:
        from agentrial.runner.adapters.smolagents import SmolAgentsAdapter

        mock_agent = MagicMock()
        mock_agent.run.return_value = "Final answer"
        mock_agent.logs = [
            {
                "llm_output": "Let me search for that.",
                "tool_call": {
                    "tool_name": "web_search",
                    "tool_arguments": {"query": "test"},
                },
            },
            {
                "observation": "Found 5 results",
            },
        ]
        mock_agent.monitor = MagicMock(
            total_input_token_count=200,
            total_output_token_count=100,
        )

        adapter = SmolAgentsAdapter(mock_agent)
        result = adapter(AgentInput(query="search something"))

        assert result.output == "Final answer"
        assert result.success
        # llm_output + tool_call + observation = 3 steps
        assert len(result.steps) == 3
        assert result.steps[0].step_type == StepType.LLM_CALL
        assert result.steps[1].step_type == StepType.TOOL_CALL
        assert result.steps[1].name == "web_search"
        assert result.steps[2].step_type == StepType.OBSERVATION

    def test_adapter_processes_object_logs(self) -> None:
        from agentrial.runner.adapters.smolagents import SmolAgentsAdapter

        @dataclass
        class MockToolCall:
            name: str = "calculator"
            arguments: dict = field(default_factory=lambda: {"expr": "2+2"})

        @dataclass
        class MockStep:
            tool_calls: list = field(
                default_factory=lambda: [MockToolCall()]
            )
            llm_output: str = "Computing..."
            observations: str = "Result: 4"

        mock_agent = MagicMock()
        mock_agent.run.return_value = "4"
        mock_agent.logs = [MockStep()]
        mock_agent.monitor = None

        adapter = SmolAgentsAdapter(mock_agent)
        result = adapter(AgentInput(query="2+2"))

        assert result.output == "4"
        assert result.success
        # tool_call + llm_output + observation = 3 steps
        assert len(result.steps) == 3

    def test_adapter_handles_error(self) -> None:
        from agentrial.runner.adapters.smolagents import SmolAgentsAdapter

        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("Model error")

        adapter = SmolAgentsAdapter(mock_agent)
        result = adapter(AgentInput(query="test"))

        assert not result.success
        assert "Model error" in result.error

    def test_adapter_empty_logs(self) -> None:
        from agentrial.runner.adapters.smolagents import SmolAgentsAdapter

        mock_agent = MagicMock()
        mock_agent.run.return_value = "Simple answer"
        mock_agent.logs = []
        mock_agent.monitor = None

        adapter = SmolAgentsAdapter(mock_agent)
        result = adapter(AgentInput(query="test"))

        assert result.output == "Simple answer"
        assert result.success
        assert len(result.steps) == 0


# --- Lazy import tests ---


class TestLazyImports:
    """Test that __init__.py lazy imports work correctly."""

    def test_lazy_import_nonexistent_attr(self) -> None:
        import agentrial.runner.adapters as adapters_mod

        with pytest.raises(AttributeError, match="no attribute"):
            _ = adapters_mod.nonexistent_attribute

    def test_direct_imports_available(self) -> None:
        from agentrial.runner.adapters import BaseAdapter, wrap_langgraph_agent

        assert BaseAdapter is not None
        assert wrap_langgraph_agent is not None

    def test_langgraph_adapter_inherits_base(self) -> None:
        """LangGraphAdapter should be an instance of BaseAdapter (H6)."""
        from agentrial.runner.adapters.base import BaseAdapter
        from agentrial.runner.adapters.langgraph import LangGraphAdapter

        adapter = LangGraphAdapter(graph=MagicMock())
        assert isinstance(adapter, BaseAdapter)
