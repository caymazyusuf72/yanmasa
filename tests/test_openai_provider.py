"""OpenAI ve OpenAI-Uyumlu Sağlayıcı Testleri."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from backend import config
from backend.agent.loop import Agent, Turn
from backend.agent.provider import (
    AnthropicProvider,
    ModelResponse,
    OpenAIProvider,
    TextBlock,
    ToolUseBlock,
)
from backend.agent.tools import COMPUTER_TOOLS, to_openai_tool
from backend.computer.displays import Display, DisplayMap


class TestConfigOpenAI:
    def test_config_auto_detect_anthropic(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        cfg = config.Config.load()
        assert cfg.llm_provider == "anthropic"
        assert cfg.anthropic_api_key == "sk-ant-test-key"
        assert cfg.active_model == "claude-opus-5"

    def test_config_auto_detect_openai(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-openai-test-key")
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        cfg = config.Config.load()
        assert cfg.llm_provider == "openai"
        assert cfg.openai_api_key == "sk-proj-openai-test-key"
        assert cfg.active_model == "gpt-4o"

    def test_config_explicit_provider_openai(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-key")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_MODEL", "deepseek-chat")

        cfg = config.Config.load()
        assert cfg.llm_provider == "openai"
        assert cfg.active_model == "deepseek-chat"

    def test_config_openai_base_url_local(self, monkeypatch):
        # Yerel sunucu senaryosu: API anahtarı yok, sadece base_url var (Ollama, LM Studio)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_MODEL", "qwen2.5-coder:32b")

        cfg = config.Config.load()
        assert cfg.llm_provider == "openai"
        assert cfg.openai_base_url == "http://localhost:11434/v1"
        assert cfg.openai_api_key == "dummy-key"
        assert cfg.active_model == "qwen2.5-coder:32b"

    def test_config_missing_keys_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_PROVIDER", raising=False)

        with pytest.raises(RuntimeError) as exc:
            config.Config.load()
        assert "ANTHROPIC_API_KEY is missing" in str(exc.value)


class TestOpenAITools:
    def test_to_openai_tool_conversion(self):
        tool = {
            "name": "custom_action",
            "description": "Performs custom action",
            "input_schema": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        }
        converted = to_openai_tool(tool)
        assert converted["type"] == "function"
        assert converted["function"]["name"] == "custom_action"
        assert converted["function"]["description"] == "Performs custom action"
        assert converted["function"]["parameters"] == tool["input_schema"]

    def test_computer_tools_coverage(self):
        names = {t["name"] for t in COMPUTER_TOOLS}
        expected = {
            "screenshot", "zoom", "left_click", "right_click", "middle_click",
            "double_click", "triple_click", "mouse_move", "left_mouse_down",
            "left_mouse_up", "cursor_position", "left_click_drag", "scroll",
            "type", "key", "wait",
        }
        assert expected.issubset(names)

    def test_provider_tools_wrapping(self):
        mock_client = MagicMock()
        provider = OpenAIProvider(mock_client, model="gpt-4o")
        tools = provider.tools(
            skill_tools=[{"name": "my_skill", "description": "desc", "input_schema": {}}],
            mcp_tools=[{"name": "mcp__fetch", "description": "desc", "input_schema": {}}],
        )
        assert all(t.get("type") == "function" for t in tools)
        tool_names = {t["function"]["name"] for t in tools}
        assert "screenshot" in tool_names
        assert "run_shell" in tool_names
        assert "my_skill" in tool_names
        assert "mcp__fetch" in tool_names


class TestOpenAIMessageConversion:
    def test_convert_text_and_tool_calls(self):
        mock_client = MagicMock()
        provider = OpenAIProvider(mock_client)

        messages = [
            {"role": "user", "content": "Open Notepad and type hello"},
            {
                "role": "assistant",
                "content": [
                    TextBlock(text="Opening notepad..."),
                    ToolUseBlock(id="call_1", name="launch_app", input={"target": "notepad"}),
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": "Notepad launched",
                    }
                ],
            },
        ]

        converted = provider._convert_messages(messages, system_text="You are an assistant.")
        assert len(converted) == 4
        assert converted[0] == {"role": "system", "content": "You are an assistant."}
        assert converted[1] == {"role": "user", "content": "Open Notepad and type hello"}
        assert converted[2]["role"] == "assistant"
        assert converted[2]["content"] == "Opening notepad..."
        assert len(converted[2]["tool_calls"]) == 1
        assert converted[2]["tool_calls"][0]["function"]["name"] == "launch_app"
        assert converted[3] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Notepad launched",
        }

    def test_convert_multimodal_image_result(self):
        mock_client = MagicMock()
        provider = OpenAIProvider(mock_client)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_ss",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                                },
                            }
                        ],
                    }
                ],
            }
        ]

        converted = provider._convert_messages(messages, system_text="")
        assert len(converted) == 2
        assert converted[0]["role"] == "tool"
        assert converted[0]["tool_call_id"] == "call_ss"
        assert converted[1]["role"] == "user"
        assert isinstance(converted[1]["content"], list)
        assert converted[1]["content"][0]["type"] == "image_url"
        assert "data:image/png;base64," in converted[1]["content"][0]["image_url"]["url"]


class TestOpenAIStreaming:
    def test_stream_text_reasoning_and_tools(self):
        mock_client = MagicMock()
        provider = OpenAIProvider(mock_client, model="deepseek-chat")

        # Fake stream chunks
        class FakeDelta:
            def __init__(self, content=None, reasoning_content=None, tool_calls=None):
                self.content = content
                self.reasoning_content = reasoning_content
                self.tool_calls = tool_calls

        class FakeChoice:
            def __init__(self, delta):
                self.delta = delta

        class FakeChunk:
            def __init__(self, delta):
                self.choices = [FakeChoice(delta)]

        class FakeToolCallChunk:
            def __init__(self, index, id=None, name=None, arguments=None):
                self.index = index
                self.id = id
                self.function = MagicMock()
                self.function.name = name
                self.function.arguments = arguments

        mock_client.chat.completions.create.return_value = iter([
            FakeChunk(FakeDelta(reasoning_content="Thinking about task...")),
            FakeChunk(FakeDelta(content="I will write a file.\n")),
            FakeChunk(FakeDelta(tool_calls=[
                FakeToolCallChunk(index=0, id="call_write_1", name="write_file", arguments='{"path": "test.txt", ')
            ])),
            FakeChunk(FakeDelta(tool_calls=[
                FakeToolCallChunk(index=0, id=None, name=None, arguments='"content": "Hello World"}')
            ])),
        ])

        pulses = []
        texts = []
        thoughts = []
        codes = []

        turn = Turn(
            on_pulse=lambda: pulses.append(1),
            on_text=lambda t: texts.append(t),
            on_thinking=lambda th: thoughts.append(th),
            on_kod=lambda a, y, m, d: codes.append((a, y, m, d)),
        )

        resp = provider.call_stream(
            messages=[{"role": "user", "content": "write test.txt"}],
            system_text="System instructions",
            tools=[],
            turn=turn,
        )

        assert resp.stop_reason == "tool_use"
        assert len(resp.content) == 2
        assert isinstance(resp.content[0], TextBlock)
        assert resp.content[0].text == "I will write a file.\n"
        assert isinstance(resp.content[1], ToolUseBlock)
        assert resp.content[1].id == "call_write_1"
        assert resp.content[1].name == "write_file"
        assert resp.content[1].input == {"path": "test.txt", "content": "Hello World"}

        assert "".join(texts) == "I will write a file.\n"
        assert "Thinking about task..." in "".join(thoughts)
        assert len(pulses) >= 1
        assert len(codes) >= 1


class TestAgentIntegrationWithOpenAI:
    def test_agent_create_with_openai(self, monkeypatch):
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

        cfg = config.Config.load()
        mock_capture = MagicMock()
        mock_kill = MagicMock()
        displays = DisplayMap([Display(0, 0, 0, 1920, 1080, True)])

        agent = Agent.create(cfg, displays, mock_capture, mock_kill)
        assert isinstance(agent.provider, OpenAIProvider)
        assert agent.provider.model == "gpt-4o-mini"
        assert agent.provider.client.base_url.host == "api.openai.com"
