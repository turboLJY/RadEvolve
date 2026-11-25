"""Capture real tool inputs/outputs, generated views and conversation history."""
from __future__ import annotations

import copy
import json


def state_records(context) -> dict[str, tuple[str, dict]]:
    state = context.extra.get("pipeline_state")
    store = context.extra.get("image_store")
    records = {}
    if store:
        for item in store.list_all():
            records[item["image_id"]] = ("image", item)
    if state is None:
        return records
    for item in state.observation_graph:
        node_id = item.get("observation_id") or item.get("id")
        if node_id:
            records[node_id] = ("observation", copy.deepcopy(item))
    for item in state.retrieved_evidence:
        records[item.evidence_id] = ("evidence", item.to_dict())
    if state.structured_report:
        for item in state.structured_report.findings:
            records[item.finding_id] = ("finding", item.to_dict())
        for item in state.structured_report.impressions:
            records[item.impression_id] = ("impression", item.to_dict())
    for item in state.review_issues:
        records[item.issue_id] = ("review", item.to_dict())
    return records


def referenced_ids(value, known_ids: set[str]) -> set[str]:
    """Only explicit identifier values count; free prose is not a dependency."""
    if isinstance(value, str):
        return {value} if value in known_ids else set()
    if isinstance(value, dict):
        return set().union(*(referenced_ids(v, known_ids) for v in value.values())) if value else set()
    if isinstance(value, (list, tuple)):
        return set().union(*(referenced_ids(v, known_ids) for v in value)) if value else set()
    return set()


def record_tool(context, tool_call, result, before, elapsed):
    state = context.extra.get("pipeline_state")
    if state is None:
        return
    after = state_records(context)
    changed = {key: value for key, value in after.items() if before.get(key) != value}
    def semantic_record(value):
        kind, payload = value
        payload = copy.deepcopy(payload)
        if kind in {"finding", "impression"}:
            payload.get("support", {}).pop("relations", None)
        return kind, payload

    # Recording a review is not recomputing the claim it reviewed. In particular,
    # adding support judgements must not clear an upstream invalidation or
    # replace the writing operation as the claim's producer.
    produced = {key: value for key, value in changed.items()
                if key not in before or semantic_record(before[key]) != semantic_record(value)}
    if not result.is_error:
        for node_id in result.metadata.get("produced_ids", []):
            if node_id not in after:
                raise ValueError(f"Tool declared an unknown produced identifier: {node_id}")
            produced[node_id] = after[node_id]
    graph = state.dependency_graph
    for node_id, (kind, payload) in after.items():
        if node_id in produced or node_id not in graph.nodes:
            graph.add_node(node_id, kind, payload)
    op_id = context.extra.get("operation_id") or f"OP_{context.session_id}_{len(state.execution_trace) + 1:06d}"
    consumed = referenced_ids(tool_call.get("tool_input", {}), set(before)) - set(produced)
    # Source images of crops/previews are supplied by the tool's metadata.
    for _, payload in produced.values():
        consumed |= referenced_ids(payload.get("metadata", {}), set(before)) - set(produced)
    output_ids = sorted(produced)
    record = {
        "operation_id": op_id, "tool_call_id": tool_call.get("tool_use_id", ""),
        "stage": context.extra.get("agent_role", ""), "turn": context.turn_count,
        "tool_name": tool_call["tool_name"], "inputs": tool_call.get("tool_input", {}),
        "consumed_ids": sorted(consumed), "produced_ids": output_ids,
        "result": result.data, "metadata": result.metadata, "is_error": result.is_error,
        "elapsed_seconds": elapsed, "versions": context.extra.get("runtime_versions", {}),
    }
    graph.add_node(op_id, "tool_call", record)
    for source in consumed:
        graph.add_edge(source, op_id)
    for target in output_ids:
        # Replacing producer links keeps current dependencies free of stale
        # operation revisions; historical revisions remain in SQLite.
        graph.edges = {(a, b) for a, b in graph.edges
                       if not (b == target and graph.nodes.get(a, {}).get("kind") == "tool_call")}
        graph.add_edge(op_id, target)
    state.execution_trace.append(record)
    durable = context.extra.get("record_store")
    if durable:
        durable.append(context.session_id, op_id, "tool_call", record)
        for node_id, (kind, payload) in changed.items():
            durable.append(context.session_id, node_id, kind, payload)


def record_stage(params, loop_state, reason):
    extra = getattr(params, "_extra_context", {})
    state = extra.get("pipeline_state")
    if state is None:
        return
    stage = extra.get("agent_role", "unknown")
    record = {"stage": stage, "iteration": state.iteration_count, "reason": reason,
              "messages": [m.to_api_dict() for m in loop_state.messages],
              "turns": loop_state.turn_count, "compactions": loop_state.compaction_count}
    if not hasattr(state, "conversation_history"):
        state.conversation_history = {}
    state.conversation_history[stage] = copy.deepcopy(record)
    durable = extra.get("record_store")
    if durable:
        durable.append(state.session_id, f"STAGE_{stage}_{state.iteration_count}", "stage", record)


def record_request(params, event):
    request = event.data.get("request_record")
    if not request:
        return
    extra = getattr(params, "_extra_context", {})
    state = extra.get("pipeline_state")
    if state is not None:
        if not hasattr(state, "request_records"):
            state.request_records = []
        state.request_records.append(copy.deepcopy(request))
    durable = extra.get("record_store")
    if durable:
        durable.append(params.session_id, request["request_id"], "request", request)
