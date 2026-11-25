import json
from types import SimpleNamespace

import pytest
from PIL import Image

from src.core.provenance import DependencyGraph, SQLiteRecordStore
from src.core.tool import ToolRegistry, Tool, ToolUseContext
from src.core.types import PipelineState, StreamEvent, EventType, ToolResult
from src.core.loop import QueryParams, query_loop, _execute_tools
from src.data.image_store import SessionImageStore
from src.core.inputs import reporting_context


def test_generation_context_excludes_targets_but_keeps_available_priors():
    source = {"modality": "CT", "positive_labels": ["disease"], "reference_report": "answer",
              "image_metadata": {"slice": 2, "labels": [1]},
              "prior_examinations": [{"examination_id": "prior", "report": "Historical report"}]}
    filtered = reporting_context(source)
    assert "positive_labels" not in filtered and "reference_report" not in filtered
    assert filtered["image_metadata"] == {"slice": 2}
    assert filtered["prior_examinations"][0]["report"] == "Historical report"
    assert "age" not in filtered
    assert source["image_metadata"]["labels"] == [1]


def test_dependency_repair_replaces_stale_links_and_preserves_unaffected():
    graph = DependencyGraph()
    for key in ("old", "new", "finding", "impression", "unrelated"):
        graph.add_node(key, "observation", {})
    graph.set_dependencies("finding", ["old"])
    graph.set_dependencies("impression", ["finding"])
    assert graph.invalidate({"old"}) == {"old", "finding", "impression"}
    graph.set_dependencies("finding", ["new"])
    assert graph.descendants({"old"}) == set()
    assert graph.descendants({"new"}) == {"finding", "impression"}
    with pytest.raises(ValueError, match="cycle"):
        graph.add_edge("impression", "new")
    assert DependencyGraph.from_dict(graph.to_dict()).to_dict() == graph.to_dict()


def test_sqlite_keeps_revisions_and_rejects_invalid_json(tmp_path):
    with SQLiteRecordStore(tmp_path / "records.sqlite") as db:
        first = db.append("case", "finding", "finding", {"text": "before"})
        second = db.append("case", "finding", "finding", {"text": "after"})
        assert first != second
        assert [r["payload"]["text"] for r in db.read("case", "finding")] == ["before", "after"]
        with pytest.raises(Exception):
            db.append("case", "bad", "unknown_kind", {})
        with pytest.raises(ValueError):
            db.append("case", "bad", "finding", {"confidence": float("nan")})


class ViewTool(Tool):
    name = "make_view"
    input_schema = {"type": "object", "properties": {"size": {"type": "integer", "minimum": 1}},
                    "required": ["size"]}
    is_read_only = False
    is_concurrency_safe = True

    async def call(self, args, context):
        store = context.extra["image_store"]
        view_id = store.save(Image.new("RGB", (args["size"], 12), "red"), "derived view", self.name)
        return ToolResult(json.dumps({"image_id": view_id}), metadata={"image_id": view_id})


class ScriptedClient:
    def __init__(self):
        self.inputs = []

    async def stream(self, **kwargs):
        self.inputs.append(kwargs)
        if len(self.inputs) == 1:
            yield StreamEvent(EventType.TOOL_USE, {"tool_name": "make_view", "tool_use_id": "call_1",
                                                 "tool_input": {"size": 20}})
        else:
            yield StreamEvent(EventType.TEXT_DELTA, {"text": "done"})


@pytest.mark.asyncio
async def test_tool_image_is_seen_by_next_model_turn_and_trace_is_persisted(tmp_path):
    registry = ToolRegistry()
    registry.register(ViewTool())
    state = PipelineState(session_id="case")
    image_store = SessionImageStore("case")
    client = ScriptedClient()
    with SQLiteRecordStore(tmp_path / "records.sqlite") as db:
        params = QueryParams("inspect", "system", registry, client, session_id="case", cache_dir=str(tmp_path))
        params._extra_context = {"pipeline_state": state, "image_store": image_store,
                                 "record_store": db, "agent_role": "visual_analysis"}
        events = [event async for event in query_loop(params)]
        assert events[-1].data["reason"] == "complete"
        blocks = client.inputs[1]["messages"][-1]["content"]
        assert any(b.get("type") == "image_url" for b in blocks)
        assert state.execution_trace[0]["produced_ids"] == [image_store.list_all()[0]["image_id"]]
        assert any(r["kind"] == "tool_call" for r in db.read("case"))
        assert any(r["kind"] == "stage" for r in db.read("case"))


@pytest.mark.asyncio
async def test_invalid_tool_input_has_a_record_without_mutating_state(tmp_path):
    registry = ToolRegistry()
    registry.register(ViewTool())
    store = SessionImageStore("case")
    state = PipelineState(session_id="case")
    context = ToolUseContext(session_id="case", cache_dir=tmp_path,
                             extra={"pipeline_state": state, "image_store": store})
    results = await _execute_tools([{"tool_name": "make_view", "tool_input": {"size": -1}}], registry,
                                   context, registry.get_concurrent_safe())
    assert results[0].is_error
    assert store.count() == 0
    assert state.execution_trace[0]["is_error"]
