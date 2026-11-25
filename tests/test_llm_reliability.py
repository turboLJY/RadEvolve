from types import SimpleNamespace as NS

import pytest

from src.llm.client import LLMClient, ModelConfig, _is_retryable
from src.core.types import EventType


def chunk(text=None, calls=None, usage=None):
    return NS(usage=usage, choices=[NS(delta=NS(content=text, tool_calls=calls))])


def call(index, name=None, args=None, call_id=None):
    return NS(index=index, id=call_id, function=NS(name=name, arguments=args))


@pytest.mark.asyncio
async def test_retry_drops_partial_response_and_keeps_attempt_accounting():
    client = LLMClient(ModelConfig(max_retries=1, retry_base_delay_ms=0))
    attempts = []

    async def create(**kwargs):
        attempts.append(kwargs)
        async def response():
            if len(attempts) == 1:
                yield chunk("discard me")
                raise ConnectionError("transient")
            yield chunk("complete")
        return response()

    client.client = NS(chat=NS(completions=NS(create=create)))
    events = [e async for e in client.stream([], "system")]
    assert [e.data["text"] for e in events if e.type == EventType.TEXT_DELTA] == ["complete"]
    records = [e.data["request_record"] for e in events if "request_record" in e.data]
    assert [r["complete"] for r in records] == [False, True]
    assert all(r["cost_cny"] is None for r in records)


@pytest.mark.asyncio
async def test_interleaved_tool_arguments_are_assembled_by_index():
    client = LLMClient(ModelConfig(max_retries=0))
    async def create(**kwargs):
        async def response():
            yield chunk(calls=[call(0, "first", '{"a":', "a"), call(1, "second", '{"b":', "b")])
            yield chunk(calls=[call(1, args="2}"), call(0, args="1}")])
        return response()
    client.client = NS(chat=NS(completions=NS(create=create)))
    events = [e async for e in client.stream([], "system")]
    tools = [e.data for e in events if e.type == EventType.TOOL_USE]
    assert [t["tool_input"] for t in tools] == [{"a": 1}, {"b": 2}]


def test_clinical_content_errors_are_not_retried():
    assert not _is_retryable(ValueError("DataInspectionFailed: clinical content 500"))
    assert _is_retryable(ConnectionError("connection lost"))


def test_retry_limit_and_invalid_settings_do_not_fall_back(tmp_path):
    with pytest.raises(ValueError, match="five"):
        ModelConfig(max_retries=6)
    settings = tmp_path / "settings.yaml"
    settings.write_text("model:\n  max_retries: 6\n")
    with pytest.raises(ValueError, match="refusing"):
        ModelConfig.from_settings(settings)
