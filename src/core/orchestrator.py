"""
Multi-agent orchestrator for RadClaw.

Sequences 4 agents (Visual Analysis → Evidence Retrieval → Report Writing → Quality Review)
with a quality review feedback loop that triggers targeted re-execution.

This is the major new component not present in SearchClaw.
"""

from __future__ import annotations

import base64
import io
import logging
import inspect
from pathlib import Path
import re
import uuid
from typing import AsyncGenerator

from PIL import Image

from src.core.types import (
    AgentRole,
    ContentBlock,
    EventType,
    ImageEvidence,
    Message,
    PipelineState,
    StreamEvent,
    StructuredReport,
)
from src.core.loop import QueryParams, query_loop
from src.agents.base import build_agent
from src.data.image_store import SessionImageStore
from src.llm.client import LLMClient
from src.core.provenance import SQLiteRecordStore
from src.core.inputs import reporting_context
from src.report.grounding import claim_id, claims, integrity_errors, register_issue, render_claims
from src.evolution.runtime import load_runtime, check_runtime, verify_agent_config

logger = logging.getLogger(__name__)

AGENT_SEQUENCE = [
    AgentRole.VISUAL_ANALYSIS,
    AgentRole.EVIDENCE_RETRIEVAL,
    AgentRole.REPORT_WRITING,
    AgentRole.QUALITY_REVIEW,
]

ATTRIBUTION_TO_AGENT = {
    "visual": AgentRole.VISUAL_ANALYSIS,
    "retrieval": AgentRole.EVIDENCE_RETRIEVAL,
    "generation": AgentRole.REPORT_WRITING,
    "wording": AgentRole.REPORT_WRITING,
}


MODALITY_MAP = {
    "X-Ray": "CXR",
    "x-ray": "CXR",
    "xray": "CXR",
    "CXR": "CXR",
    "cxr": "CXR",
    "chest_ct": "chest_ct",
    "CT": "chest_ct",
    "ct": "chest_ct",
    "abdominal_ct": "abdominal_ct",
    "brain_ct": "brain_ct",
    "brain_mri": "brain_mri",
    "MRI": "brain_mri",
    "mri": "brain_mri",
}

MODALITY_DISPLAY = {
    "CXR": "chest X-ray",
    "chest_ct": "chest CT",
    "abdominal_ct": "abdominal CT",
    "brain_ct": "brain CT",
    "brain_mri": "brain MRI",
}


def _resolve_modality(exam_info: dict) -> str:
    raw = exam_info.get("modality", "")
    return MODALITY_MAP.get(raw, raw)


def _image_to_base64(image: Image.Image) -> str:
    max_dim = 1024
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        logger.info("Resized image from original to %s for LLM input", new_size)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def _determine_agents_to_run(state: PipelineState) -> list[AgentRole]:
    if state.iteration_count <= 1:
        return list(AGENT_SEQUENCE)

    agents_needed = set()
    for issue in state.review_issues:
        target = ATTRIBUTION_TO_AGENT.get(issue.attribution)
        if target:
            agents_needed.add(target)

    if not agents_needed:
        return []

    # Always run all downstream agents from the earliest needed one,
    # so that visual changes propagate through evidence → report → review.
    pipeline_without_qr = [r for r in AGENT_SEQUENCE if r != AgentRole.QUALITY_REVIEW]
    earliest_idx = min(pipeline_without_qr.index(r) for r in agents_needed)
    result = pipeline_without_qr[earliest_idx:]
    result.append(AgentRole.QUALITY_REVIEW)
    return result


def _build_agent_query(agent_role: AgentRole, state: PipelineState) -> str:
    modality_display = MODALITY_DISPLAY.get(state.modality, state.modality or "medical image")

    if agent_role == AgentRole.VISUAL_ANALYSIS:
        source_size = getattr(state.original_image, "size", None)
        return (
            f"Attached study preview ID: {state.original_image_id or state.active_image_id}. "
            f"Its stored source dimensions are {source_size} pixels (width, height); cropping coordinates "
            f"refer to stored source pixels, even when the displayed preview is resized to 1024 pixels. "
            f"Currently selected source view ID: {state.active_image_id}.\n"
            f"Analyze this {modality_display} image. Use your visual analysis tools to "
            "systematically examine the anatomy relevant to the supplied modality and sequences. "
            "For each finding, use the appropriate tools (crop_roi, zoom_and_enhance, "
            "multi_window_display, overlay_annotation) and bind evidence using link_visual_evidence. "
            "Provide a comprehensive summary of all findings."
        )

    elif agent_role == AgentRole.EVIDENCE_RETRIEVAL:
        findings = state.visual_findings_text or "No visual findings available."
        evidence_list = "\n".join(
            f"  - {e.description} (image: {e.image_id})"
            for e in state.image_evidence
        )
        return (
            f"Given these visual findings from {modality_display} analysis:\n\n"
            f"{findings}\n\n"
            f"Image evidence:\n{evidence_list}\n\n"
            f"Retrieve supporting medical evidence. Use standardize_observation for each "
            f"finding, then use concept_lookup, literature_search, and case_retrieval "
            f"to search knowledge bases. Use evidence_score_and_rank to rank results, "
            f"conflict_detection to check for contradictions, and evidence_summary to "
            f"generate a consolidated summary."
        )

    elif agent_role == AgentRole.REPORT_WRITING:
        findings = state.visual_findings_text or ""
        evidence = state.evidence_summary or ""
        img_refs = ", ".join(e.image_id for e in state.image_evidence)
        ev_refs = ", ".join(e.evidence_id for e in state.retrieved_evidence)
        conflicts = ""
        if state.conflicts:
            conflicts = "\nConflicts detected:\n" + "\n".join(
                f"  - {c.get('description', str(c))}" for c in state.conflicts[:5]
            )
        return (
            f"Generate a structured radiology report from the following analysis:\n\n"
            f"VISUAL FINDINGS:\n{findings}\n\n"
            f"EVIDENCE SUMMARY:\n{evidence}\n"
            f"{conflicts}\n\n"
            f"Available image evidence IDs: {img_refs}\n"
            f"Available retrieved evidence IDs: {ev_refs}\n\n"
        f"Use build_structured_finding for each finding, build_structured_impression "
            f"for each impression, link_support_evidence to attach evidence, then "
            f"assemble_report_object and render_natural_language."
        )

    elif agent_role == AgentRole.QUALITY_REVIEW:
        report_json = ""
        if state.structured_report:
            import json
            report_json = json.dumps(state.structured_report.to_dict(), indent=2, ensure_ascii=False)
        return (
            f"Review this radiology report for quality and consistency.\n\n"
            f"STRUCTURED REPORT:\n{report_json}\n\n"
        f"Check all four dimensions:\n"
            f"1. Visual consistency (check_visual_consistency)\n"
            f"2. Evidence consistency (check_evidence_consistency)\n"
            f"3. Internal consistency (check_internal_consistency)\n"
            f"4. Medical standard compliance (check_medical_standard)\n\n"
            f"For EACH issue found, use attribute_responsibility to assign it.\n"
            f"Record every claim review and observation omission check with record_claim_review. "
            f"The controller determines PASS from coverage, traceability and recorded defects."
        )

    return ""


def _build_rerun_query(agent_role: AgentRole, state: PipelineState) -> str:
    relevant_issues = [
        i for i in state.repair_issues or state.review_issues
        if ATTRIBUTION_TO_AGENT.get(i.attribution) == agent_role
    ]

    if not relevant_issues:
        return _build_agent_query(agent_role, state)

    feedback_text = "\n".join(
        f"  - [{i.severity}] claim={i.related_claim_id or i.omitted_observation_id}: {i.description} (suggested fix: {i.suggested_fix})"
        for i in relevant_issues
    )

    base_query = _build_agent_query(agent_role, state)

    return (
        f"REVISION REQUIRED\n\n"
        f"The quality review found these issues with your previous output:\n"
        f"{feedback_text}\n\n"
        f"Please address these specific issues. Affected dependency IDs: {sorted(state.affected_node_ids)}. "
        f"Preserve content and evidence of all other claims; replace affected claims by their existing IDs.\n\n"
        f"---\n\n{base_query}"
    )


def _prepare_remediation(state: PipelineState) -> None:
    """Invalidate modified roots and descendants, retaining unrelated claims."""
    graph = state.dependency_graph
    roots = set()
    state.repair_issues = list(state.review_issues)
    for issue in state.repair_issues:
        identifier = issue.implicated_operation or issue.omitted_observation_id or issue.related_claim_id
        if identifier:
            roots.add(identifier)
            if not issue.implicated_operation and issue.attribution in {"visual", "retrieval"}:
                kinds = {"observation", "image_evidence", "image", "image_operation"} if issue.attribution == "visual" else {"evidence", "retrieved_evidence", "retrieval"}
                roots.update(node for node in graph.ancestors({identifier}) if graph.nodes[node]["kind"] in kinds)
        else:
            # A report-level failure has no narrower dependency scope.
            roots.update(claim_id(c) for c in claims(state))
            if issue.attribution == "visual":
                roots.update(o["observation_id"] for o in state.observation_graph if o.get("observation_id"))
    state.affected_node_ids = graph.invalidate(roots)
    state.repaired_node_ids.clear()
    state.reviewed_claim_ids.difference_update(state.affected_node_ids)
    state.reviewed_observation_ids.difference_update(state.affected_node_ids)
    for identifier in state.affected_node_ids:
        state.claim_reviews.pop(identifier, None)
    state.review_completed = False
    state.review_passed = False
    state.review_issues = []


def _finalize_review(state: PipelineState) -> None:
    for identifier, category, description in integrity_errors(state):
        register_issue(state, identifier, category, description)
    expected = {claim_id(c) for c in claims(state)}
    observations = {o["observation_id"] for o in state.observation_graph if o.get("observation_id")}
    missing = expected - state.reviewed_claim_ids
    for identifier in missing:
        register_issue(state, identifier, "review_incomplete", "Claim has no complete four-dimension review record")
    if observations - state.reviewed_observation_ids:
        register_issue(state, "", "review_incomplete", "Observations have not all been compared with report claims for omissions")
    invalid_claims = expected & state.dependency_graph.invalidated
    for identifier in invalid_claims:
        register_issue(state, identifier, "traceability_failure", "Affected claim was not recomputed after upstream content changed")
    if state.stage_failures:
        register_issue(state, "", "execution_incomplete", "One or more required reporting stages failed")
    state.review_completed = bool(expected) and not missing and observations <= state.reviewed_observation_ids
    blocking = any(issue.severity in {"high", "critical"} or issue.issue_type in {
        "traceability_failure", "internal_inconsistency", "internal_contradiction", "review_incomplete", "execution_incomplete"
    } for issue in state.review_issues)
    state.review_passed = state.review_completed and not blocking and bool(state.structured_report)
    state.review_status = "PASS" if state.review_passed else "REVISE"


async def run_pipeline(
    image: Image.Image,
    exam_info: dict,
    settings: dict,
    llm_client: LLMClient,
    *,
    initial_state: PipelineState | None = None,
    image_store: SessionImageStore | None = None,
    start_stage: AgentRole | str | None = None,
    runtime_bundle=None,
    checkpoint_callback=None,
) -> AsyncGenerator[StreamEvent, str | None]:
    """Run or resume a case with fixed runtime versions and dependency-scoped repair."""
    exam_info = reporting_context(exam_info)
    session_id = initial_state.session_id if initial_state else str(uuid.uuid4())[:8]
    modality = _resolve_modality(exam_info)
    max_passes = min(3, max(1, int(settings.get("pipeline", {}).get("max_iterations", 3))))
    store = image_store or SessionImageStore(session_id, max_views=settings.get("limits", {}).get("max_image_views", 32))
    if initial_state is None:
        original_id = store.save(image, f"Original {MODALITY_DISPLAY.get(modality, modality)}", "input", metadata=exam_info.get("preview_metadata", {}))
        state = PipelineState(session_id=session_id, original_image=image, exam_info=dict(exam_info),
                              modality=modality, max_iterations=max_passes, active_image_id=original_id, original_image_id=original_id)
    else:
        state = initial_state
        state.exam_info = reporting_context(state.exam_info)
        state.max_iterations = max_passes
    bundle = runtime_bundle or load_runtime(settings)
    check_runtime(bundle, settings, llm_client)
    state.versions = dict(bundle.versions)
    database = Path(settings.get("provenance", {}).get("database") or Path(settings.get("data", {}).get("output_dir", "./output")) / "records.sqlite")
    record_store = SQLiteRecordStore(database)
    resumed = initial_state is not None and start_stage is not None
    resume_role = AgentRole(start_stage) if start_stage is not None else None
    try:
        yield StreamEvent(type=EventType.STATUS, data={"message": f"Pipeline started (session: {session_id})", "versions": state.versions})
        while state.iteration_count < state.max_iterations or resumed:
            if resumed:
                agents_to_run = AGENT_SEQUENCE[AGENT_SEQUENCE.index(resume_role):]
                if state.iteration_count == 0:
                    state.iteration_count = 1
                resumed = False
            else:
                state.iteration_count += 1
                agents_to_run = _determine_agents_to_run(state)
                if state.iteration_count > 1:
                    _prepare_remediation(state)
            if not agents_to_run:
                break
            state.review_passed = False
            state.review_status = "REVISE"
            yield StreamEvent(type=EventType.PIPELINE_UPDATE, data={"iteration": state.iteration_count, "agents": [a.value for a in agents_to_run], "affected_node_ids": sorted(state.affected_node_ids)})
            for agent_role in agents_to_run:
                if checkpoint_callback is not None:
                    callback_result = checkpoint_callback(agent_role.value, state, store)
                    if inspect.isawaitable(callback_result):
                        await callback_result
                yield StreamEvent(type=EventType.AGENT_START, data={"agent": agent_role.value, "iteration": state.iteration_count})
                agent_config = build_agent(agent_role, settings, state)
                verify_agent_config(bundle, settings, agent_config, state)
                runtime_instructions = bundle.stage_instructions(agent_role.value, state, [t.name for t in agent_config.tool_registry.all_tools()])
                if runtime_instructions:
                    agent_config.system_prompt += "\n\n" + runtime_instructions
                query = _build_agent_query(agent_role, state) if state.iteration_count == 1 else _build_rerun_query(agent_role, state)
                images = None
                if agent_role in (AgentRole.VISUAL_ANALYSIS, AgentRole.REPORT_WRITING, AgentRole.QUALITY_REVIEW):
                    images = [{"url": "data:image/png;base64," + _image_to_base64(image)}]
                if agent_role == AgentRole.QUALITY_REVIEW:
                    # Each stage has its own conversation: explicitly expose the
                    # actual linked views, including slices from volumetric input.
                    original_ids = {item["image_id"] for item in store.list_all() if item.get("tool_chain") == ["input"]}
                    linked_views = sorted({e.image_id for e in state.image_evidence} - original_ids)
                    for view_id in linked_views:
                        encoded = store.to_base64(view_id)
                        if encoded:
                            images.append({"url": "data:image/png;base64," + encoded})
                    query += "\nAttached images in order: first the study preview " + (state.original_image_id or "original") + "; then linked views " + ", ".join(view_id for view_id in linked_views if store.get(view_id) is not None)
                params = QueryParams(query=query, system_prompt=agent_config.system_prompt,
                                     tool_registry=agent_config.tool_registry, llm_client=llm_client,
                                     max_turns=agent_config.max_turns, session_id=session_id,
                                     hook_engine=agent_config.hook_engine, temperature=agent_config.temperature,
                                     max_tokens=agent_config.max_tokens, images=images,
                                     compact_threshold_tokens=settings.get("limits", {}).get("compact_threshold_tokens", 50000))
                params._extra_context = {"pipeline_state": state, "image_store": store, "settings": settings,
                                         "agent_role": agent_role.value, "runtime_versions": state.versions,
                                         "record_store": record_store,
                                         "knowledge_dir": settings.get("data", {}).get("knowledge_dir", "./data/knowledge")}
                text_parts = []
                done = None
                async for event in query_loop(params):
                    yield event
                    if event.type == EventType.TEXT_DELTA:
                        text_parts.append(event.data.get("text", ""))
                    elif event.type == EventType.DONE:
                        done = event.data
                if done is None or done.get("reason") == "error":
                    state.stage_failures.append({"agent": agent_role.value, "iteration": state.iteration_count, "reason": (done or {}).get("reason", "incomplete")})
                agent_text = (done or {}).get("final_text") or "".join(text_parts)
                _update_pipeline_state(state, agent_role, agent_text)
                record_store.append(session_id, f"STAGE_{state.iteration_count}_{agent_role.value}", "stage", {"agent": agent_role.value, "completion": done or {}, "state": state.to_dict()})
                yield StreamEvent(type=EventType.AGENT_END, data={"agent": agent_role.value, "iteration": state.iteration_count})
            _finalize_review(state)
            state.review_history.append({"iteration": state.iteration_count, "status": state.review_status,
                                         "issues": [i.to_dict() for i in state.review_issues], "claim_reviews": dict(state.claim_reviews)})
            if state.review_passed:
                break
            if state.iteration_count < state.max_iterations:
                yield StreamEvent(type=EventType.STATUS, data={"message": f"Review requires remediation of {len(state.review_issues)} issue(s)."})
        _finalize_review(state)
        if state.structured_report:
            render_claims(state)
        result_data = state.to_dict()
        result_data.update({"iterations": state.iteration_count, "natural_language_report": state.natural_language_report,
                            "remaining_issues": [i.to_dict() for i in state.review_issues],
                            "image_evidence": [e.to_dict() for e in state.image_evidence],
                            "retrieved_evidence": [e.to_dict() for e in state.retrieved_evidence],
                            "record_database": str(database), "request_records": state.request_records,
                            "completed": not state.stage_failures})
        images_data = {}
        for item in store.list_all():
            b64 = store.to_base64(item["image_id"])
            if b64:
                images_data[item["image_id"]] = {"base64": b64, "description": item["description"],
                                               "tool_chain": item["tool_chain"], "size": list(item["size"]), "metadata": item.get("metadata", {})}
        result_data["images"] = images_data
        record_store.append(session_id, "RESULT", "result", {k: v for k, v in result_data.items() if k != "images"})
        yield StreamEvent(type=EventType.DONE, data=result_data)
    finally:
        record_store.close()


def _update_pipeline_state(state: PipelineState, agent_role: AgentRole, agent_text: str) -> None:
    state.agent_run_log.append({
        "agent": agent_role.value,
        "iteration": state.iteration_count,
        "output_length": len(agent_text),
    })

    if agent_role == AgentRole.VISUAL_ANALYSIS:
        state.visual_findings_text = agent_text

    elif agent_role == AgentRole.EVIDENCE_RETRIEVAL:
        state.evidence_summary = agent_text if agent_text.strip() else state.evidence_summary

    elif agent_role == AgentRole.REPORT_WRITING:
        if state.structured_report:
            render_claims(state)

    elif agent_role == AgentRole.QUALITY_REVIEW:
        _finalize_review(state)
