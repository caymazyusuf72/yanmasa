"""LLM Sağlayıcıları — Anthropic ve OpenAI / OpenAI-Uyumlu API'ler.

Tek bir arayüz arkasında iki sağlayıcı:
1. `AnthropicProvider`: Claude Opus 5, prompt caching, computer toolset.
2. `OpenAIProvider`: OpenAI (GPT-4o vb.), OpenRouter, Ollama, LM Studio,
   vLLM, DeepSeek, Groq gibi OpenAI uyumlu tüm servisler.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic
import openai

from .. import config
from .akankod import AkanKod
from .tools import COMPUTER_TOOLS, CUSTOM_TOOLS, to_openai_tool

PULSE_MIN_GAP = 0.05


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"
    toolset_name: str | None = None


@dataclass
class ModelResponse:
    content: list[TextBlock | ToolUseBlock]
    stop_reason: str = "end_turn"  # "tool_use" | "end_turn" | "refusal" | "stop"


class BaseProvider(ABC):
    """LLM Sağlayıcısı temel sınıfı."""

    @abstractmethod
    def tools(self, skill_tools: list[dict[str, Any]], mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Sağlayıcıya iletilecek araç listesi."""

    @abstractmethod
    def call_stream(
        self,
        messages: list[dict[str, Any]],
        system_text: str,
        tools: list[dict[str, Any]],
        turn: Any,
        effort: str = config.EFFORT,
    ) -> ModelResponse:
        """Modeli akışla çağırır ve standart `ModelResponse` döndürür."""


class AnthropicProvider(BaseProvider):
    """Anthropic Claude sağlayıcısı."""

    def __init__(self, client: anthropic.Anthropic, model: str = "claude-opus-5") -> None:
        self.client = client
        self.model = model

    def tools(self, skill_tools: list[dict[str, Any]], mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        static = [
            {"type": "computer_toolset_20260801"},
            *CUSTOM_TOOLS[:-1],
            {**CUSTOM_TOOLS[-1], "cache_control": {"type": "ephemeral"}},
        ]
        return [*static, *skill_tools, *mcp_tools]

    def call_stream(
        self,
        messages: list[dict[str, Any]],
        system_text: str,
        tools: list[dict[str, Any]],
        turn: Any,
        effort: str = config.EFFORT,
    ) -> ModelResponse:
        thinking_buffer: list[str] = []
        akan = AkanKod()
        system_blocks = [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        with self.client.messages.stream(
            model=self.model,
            max_tokens=config.MAX_TOKENS,
            system=system_blocks,
            tools=tools,
            messages=messages,
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": effort},
        ) as stream:
            son_nabiz = 0.0
            for event in stream:
                if event.type == "content_block_start":
                    blok = getattr(event, "content_block", None)
                    if getattr(blok, "type", "") == "tool_use":
                        akan.basla(getattr(blok, "name", ""))
                elif event.type == "content_block_delta":
                    delta = event.delta
                    simdi = time.monotonic()
                    if simdi - son_nabiz > PULSE_MIN_GAP:
                        son_nabiz = simdi
                        turn.on_pulse()
                    if delta.type == "text_delta":
                        turn.on_text(delta.text)
                    elif delta.type == "thinking_delta":
                        thinking_buffer.append(delta.thinking)
                    elif delta.type == "input_json_delta":
                        if akan.besle(delta.partial_json):
                            turn.on_kod(akan.arac, akan.yol, akan.metin, False)
                elif event.type == "content_block_stop":
                    if akan.etkin:
                        turn.on_kod(akan.arac, akan.yol, akan.metin, True)
                        akan.dur()
                    if thinking_buffer:
                        turn.on_thinking("".join(thinking_buffer))
                        thinking_buffer.clear()

            final = stream.get_final_message()
            blocks: list[TextBlock | ToolUseBlock] = []
            for b in final.content:
                if b.type == "text":
                    blocks.append(TextBlock(text=b.text))
                elif b.type == "tool_use":
                    blocks.append(
                        ToolUseBlock(
                            id=b.id,
                            name=b.name,
                            input=dict(b.input or {}),
                            toolset_name=getattr(b, "toolset_name", None),
                        )
                    )
            return ModelResponse(content=blocks, stop_reason=final.stop_reason)


class OpenAIProvider(BaseProvider):
    """OpenAI ve OpenAI-Uyumlu (OpenRouter, Ollama, LM Studio, vLLM, DeepSeek vb.) sağlayıcı."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str = "gpt-4o",
    ) -> None:
        self.client = client
        self.model = model

    def tools(self, skill_tools: list[dict[str, Any]], mcp_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # OpenAI modellerine hem bilgisayar hem özel araçlar standart function formatında verilir
        all_raw = [*COMPUTER_TOOLS, *CUSTOM_TOOLS, *skill_tools, *mcp_tools]
        return [to_openai_tool(t) for t in all_raw]

    def _convert_messages(self, messages: list[dict[str, Any]], system_text: str) -> list[dict[str, Any]]:
        """Yan Masa mesaj geçmişini standart OpenAI mesaj listesine dönüştürür."""
        out: list[dict[str, Any]] = []
        if system_text:
            out.append({"role": "system", "content": system_text})

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user":
                if isinstance(content, str):
                    out.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    tool_results: list[dict[str, Any]] = []
                    extra_texts: list[str] = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "tool_result":
                            tool_results.append(item)
                        elif isinstance(item, dict) and item.get("type") == "text":
                            extra_texts.append(str(item.get("text", "")))
                        elif isinstance(item, str):
                            extra_texts.append(item)

                    if tool_results:
                        for tr in tool_results:
                            call_id = str(tr.get("tool_use_id") or "call_unknown")
                            raw = tr.get("content", "")
                            if isinstance(raw, list):
                                # Görsel içeren çıktı (screenshot, zoom vb.)
                                img_urls: list[str] = []
                                txt_parts: list[str] = []
                                for part in raw:
                                    if isinstance(part, dict) and part.get("type") == "image":
                                        src = part.get("source", {})
                                        media_type = src.get("media_type", "image/png")
                                        b64 = src.get("data", "")
                                        if b64:
                                            img_urls.append(f"data:{media_type};base64,{b64}")
                                    elif isinstance(part, dict) and part.get("type") == "text":
                                        txt_parts.append(str(part.get("text", "")))
                                    elif isinstance(part, str):
                                        txt_parts.append(part)

                                tr_text = " ".join(txt_parts) or "[Screenshot captured]"
                                out.append({"role": "tool", "tool_call_id": call_id, "content": tr_text})
                                if img_urls:
                                    user_content: list[dict[str, Any]] = []
                                    for u in img_urls:
                                        user_content.append({"type": "image_url", "image_url": {"url": u}})
                                    out.append({"role": "user", "content": user_content})
                            else:
                                out.append({"role": "tool", "tool_call_id": call_id, "content": str(raw)})

                    if extra_texts:
                        out.append({"role": "user", "content": "\n".join(extra_texts)})

            elif role == "assistant":
                if isinstance(content, str):
                    out.append({"role": "assistant", "content": content})
                elif isinstance(content, list):
                    text_parts: list[str] = []
                    tool_calls: list[dict[str, Any]] = []
                    for b in content:
                        b_type = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else "")
                        if b_type == "text":
                            t = getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else "")
                            if t:
                                text_parts.append(t)
                        elif b_type == "tool_use":
                            c_id = getattr(b, "id", None) or (b.get("id") if isinstance(b, dict) else "")
                            name = getattr(b, "name", None) or (b.get("name") if isinstance(b, dict) else "")
                            inp = getattr(b, "input", None) or (b.get("input") if isinstance(b, dict) else {})
                            tool_calls.append(
                                {
                                    "id": c_id or f"call_{len(tool_calls)}",
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(inp, ensure_ascii=False)
                                        if isinstance(inp, dict)
                                        else str(inp),
                                    },
                                }
                            )
                    asst_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": "".join(text_parts) if text_parts else None,
                    }
                    if tool_calls:
                        asst_msg["tool_calls"] = tool_calls
                    out.append(asst_msg)

        return out

    def call_stream(
        self,
        messages: list[dict[str, Any]],
        system_text: str,
        tools: list[dict[str, Any]],
        turn: Any,
        effort: str = config.EFFORT,
    ) -> ModelResponse:
        openai_messages = self._convert_messages(messages, system_text)
        akan = AkanKod()
        son_nabiz = 0.0

        # Birikim tamponları
        text_buffer: list[str] = []
        thinking_buffer: list[str] = []
        tool_calls_dict: dict[int, dict[str, Any]] = {}
        in_think_tag = False

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools

        response_stream = self.client.chat.completions.create(**kwargs)

        for chunk in response_stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            # 1. Nabız
            simdi = time.monotonic()
            if simdi - son_nabiz > PULSE_MIN_GAP:
                son_nabiz = simdi
                turn.on_pulse()

            # 2. Düşünce (Reasoning Delta / reasoning_content - DeepSeek / OpenRouter / OpenAI)
            reasoning = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if reasoning:
                thinking_buffer.append(reasoning)

            # 3. Metin Akışı & <think> tag ayrıştırma
            content = delta.content
            if content:
                if "<think>" in content:
                    in_think_tag = True
                    parts = content.split("<think>", 1)
                    if parts[0]:
                        turn.on_text(parts[0])
                        text_buffer.append(parts[0])
                    content = parts[1]

                if in_think_tag:
                    if "</think>" in content:
                        parts = content.split("</think>", 1)
                        thinking_buffer.append(parts[0])
                        in_think_tag = False
                        if parts[1]:
                            turn.on_text(parts[1])
                            text_buffer.append(parts[1])
                    else:
                        thinking_buffer.append(content)
                else:
                    turn.on_text(content)
                    text_buffer.append(content)

            # 4. Araç Çağrısı Akışı
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_dict:
                        tool_calls_dict[idx] = {
                            "id": tc.id or f"call_{idx}_{int(time.time()*1000)}",
                            "name": "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_calls_dict[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls_dict[idx]["name"] += tc.function.name
                        akan.basla(tool_calls_dict[idx]["name"])
                    if tc.function and tc.function.arguments:
                        chunk_arg = tc.function.arguments
                        tool_calls_dict[idx]["arguments"] += chunk_arg
                        if akan.besle(chunk_arg):
                            turn.on_kod(akan.arac, akan.yol, akan.metin, False)

        # Akış bitti
        if akan.etkin:
            turn.on_kod(akan.arac, akan.yol, akan.metin, True)
            akan.dur()
        if thinking_buffer:
            turn.on_thinking("".join(thinking_buffer))

        # ModelResponse oluştur
        blocks: list[TextBlock | ToolUseBlock] = []
        full_text = "".join(text_buffer)
        if full_text:
            blocks.append(TextBlock(text=full_text))

        for idx in sorted(tool_calls_dict.keys()):
            tc_data = tool_calls_dict[idx]
            arg_str = tc_data["arguments"]
            try:
                parsed_args = json.loads(arg_str) if arg_str.strip() else {}
            except Exception:
                parsed_args = {"raw_arguments": arg_str}

            blocks.append(
                ToolUseBlock(
                    id=tc_data["id"],
                    name=tc_data["name"],
                    input=parsed_args if isinstance(parsed_args, dict) else {"input": parsed_args},
                )
            )

        has_tools = any(isinstance(b, ToolUseBlock) for b in blocks)
        stop_reason = "tool_use" if has_tools else "end_turn"
        return ModelResponse(content=blocks, stop_reason=stop_reason)
