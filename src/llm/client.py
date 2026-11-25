"""
LLM client for RadClaw using vLLM's OpenAI-compatible API.
Replaces SearchClaw's litellm wrapper with direct openai SDK calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import os
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import openai

from src.core.types import EventType, StreamEvent

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    model_name: str = "qwen3.6-flash"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"
    max_tokens: int = 8192
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 20
    presence_penalty: float = 1.5
    enable_thinking: bool = True
    max_retries: int = 5
    retry_base_delay_ms: int = 500
    input_cny_per_million: float | None = None
    output_cny_per_million: float | None = None

    def __post_init__(self):
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int) or not 0 <= self.max_retries <= 5:
            raise ValueError("The execution protocol permits zero to five transient retries")
        if self.max_tokens <= 0 or self.retry_base_delay_ms < 0:
            raise ValueError("Token budget must be positive and retry delay nonnegative")
        import math
        for value in (self.input_cny_per_million, self.output_cny_per_million):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError("Configured tariffs must be finite and nonnegative")

    @classmethod
    def from_settings(cls, settings_path: str | Path = "config/settings.yaml") -> ModelConfig:
        path = Path(settings_path)
        if not path.exists():
            return cls()
        try:
            import yaml
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            m = data.get("model", {})
            return cls(
                model_name=m.get("path", cls.model_name),
                base_url=m.get("base_url", cls.base_url),
                api_key=os.environ.get("RADEVOLVE_API_KEY", m.get("api_key", cls.api_key)),
                input_cny_per_million=data.get("accounting", {}).get("input_cny_per_million"),
                output_cny_per_million=data.get("accounting", {}).get("output_cny_per_million"),
                max_tokens=int(m.get("max_tokens", cls.max_tokens)),
                temperature=float(m.get("temperature", cls.temperature)),
                top_p=float(m.get("top_p", cls.top_p)),
                top_k=int(m.get("top_k", cls.top_k)),
                presence_penalty=float(m.get("presence_penalty", cls.presence_penalty)),
                enable_thinking=m.get("enable_thinking", cls.enable_thinking),
                max_retries=int(m.get("max_retries", cls.max_retries)),
                retry_base_delay_ms=int(m.get("retry_base_delay_ms", cls.retry_base_delay_ms)),
            )
        except Exception as e:
            raise ValueError(f"Invalid model settings in {path}; refusing to substitute another configuration") from e


def _is_retryable(error: Exception) -> bool:
    # Clinical content or content-filter errors are not transient request errors.
    status = getattr(error, "status_code", None)
    if status is not None:
        return status in {408, 409, 429} or 500 <= status < 600
    return isinstance(error, (openai.APIConnectionError, openai.APITimeoutError,
                              TimeoutError, ConnectionError))


class LLMClient:
    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()
        self.client = openai.AsyncOpenAI(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            max_retries=0,  # The explicit loop below owns the five-retry limit.
        )

    async def stream(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        kwargs = {
            "model": self.config.model_name,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "max_completion_tokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
        request_id = "REQ_" + uuid.uuid4().hex
        last_error = None
        for attempt in range(self.config.max_retries + 1):
            started = time.monotonic()
            usage = None
            try:
                response = await self.client.chat.completions.create(**kwargs)
                # Buffer a single attempt: failed streams must not append partial
                # claims/tool calls to the following retry's response.
                text_parts = []
                calls = {}
                async for chunk in response:
                    if getattr(chunk, "usage", None) is not None:
                        u = chunk.usage
                        usage = u.model_dump() if hasattr(u, "model_dump") else dict(u)
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        text_parts.append(delta.content)
                    for tc in delta.tool_calls or []:
                        call = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            call["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                call["name"] += tc.function.name
                            if tc.function.arguments:
                                call["arguments"] += tc.function.arguments
                events = []
                for idx, call in sorted(calls.items()):
                    args = json.loads(call["arguments"] or "{}")
                    if not isinstance(args, dict) or not call["name"]:
                        raise ValueError("Invalid structured tool call")
                    events.append(StreamEvent(type=EventType.TOOL_USE, data={
                        "tool_use_id": call["id"] or f"{request_id}_tool_{idx}",
                        "tool_name": call["name"], "tool_input": args,
                    }))
                yield self._request_event(request_id, attempt, started, usage, True)
                for part in text_parts:
                    yield StreamEvent(type=EventType.TEXT_DELTA, data={"text": part})
                for event in events:
                    yield event
                return
            except Exception as exc:
                last_error = exc
                yield self._request_event(request_id, attempt, started, usage, False,
                                          type(exc).__name__)
                if not _is_retryable(exc) or attempt >= self.config.max_retries:
                    break
                delay = self.config.retry_base_delay_ms / 1000 * 2 ** attempt
                yield StreamEvent(type=EventType.STATUS,
                                  data={"message": f"Transient API failure; retry {attempt + 1} in {delay:g}s"})
                await asyncio.sleep(delay)
        yield StreamEvent(type=EventType.ERROR,
                          data={"message": f"LLM request failed: {type(last_error).__name__}: {last_error}"})

    def _request_event(self, request_id, attempt, started, usage, complete, error=None):
        cost = None
        if usage is not None and self.config.input_cny_per_million is not None and self.config.output_cny_per_million is not None:
            cost = (usage.get("prompt_tokens", 0) * self.config.input_cny_per_million
                    + usage.get("completion_tokens", 0) * self.config.output_cny_per_million) / 1_000_000
        record = {"request_id": f"{request_id}_{attempt}", "attempt": attempt,
                  "model": self.config.model_name, "complete": complete,
                  "usage": usage, "cost_cny": cost, "error_type": error,
                  "elapsed_seconds": time.monotonic() - started}
        return StreamEvent(type=EventType.STATUS, data={"request_record": record})

    async def side_query(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int = 2048,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                temperature=0.7,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.warning(f"Side query failed: {e}")
            return ""


_shared_client: LLMClient | None = None


def set_shared_client(client: LLMClient) -> None:
    global _shared_client
    _shared_client = client


def get_shared_client() -> LLMClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = LLMClient()
    return _shared_client
