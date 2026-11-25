"""Regression tests for evidence-bound claims and local remediation (no model calls)."""
import copy
from types import SimpleNamespace

import pytest
from PIL import Image

from src.core.types import (AgentRole, EventType, Finding, ImageEvidence, Impression,
                            PipelineState, RetrievedEvidence, ReviewIssue, StreamEvent,
                            StructuredReport, SupportEvidence)
from src.core.tool import ToolRegistry, ToolUseContext
from src.core.orchestrator import (_build_rerun_query, _finalize_review, _prepare_remediation,
                                   _update_pipeline_state, run_pipeline)
from src.report.grounding import integrity_errors, render_claims, sync_claim_graph
from src.tools.report.build_finding import BuildFindingTool
from src.tools.report.build_impression import BuildImpressionTool
from src.tools.report.link_support import LinkSupportTool
from src.tools.review.claim_review import RecordClaimReviewTool


def bound_state():
    state = PipelineState(session_id="test", iteration_count=1)
    first = ImageEvidence("IMG_A", "Right lung view", finding_description="Right opacity")
    other = ImageEvidence("IMG_B", "Left pleural view", finding_description="No left effusion")
    state.image_evidence = [first, other]
    state.structured_report = StructuredReport(findings=[
        Finding("F001", "Right lung opacity.", anatomy="right lung", support=SupportEvidence(image_evidence=[first]), laterality="right", polarity="positive"),
        Finding("F002", "No left pleural effusion.", anatomy="left pleura", type="normal", support=SupportEvidence(image_evidence=[other]), polarity="negative"),
    ], impressions=[Impression("I001", "Right lung opacity.", linked_findings=["F001"])])
    for claim in state.structured_report.findings + state.structured_report.impressions:
        sync_claim_graph(state, claim)
    return state


def context(state, tmp_path):
    return ToolUseContext(session_id="test", cache_dir=tmp_path / "cache", extra={"pipeline_state": state})


def review(identifier, evidence_ids):
    return {"claim_id": identifier, "visual_support": True, "evidence_applicability": True,
            "findings_impression_consistency": True, "reporting_conventions": True,
            "rationale": "Finding agrees with the supplied evidence.",
            "evidence_relations": [{"evidence_id": i, "relation": "fully_supported", "rationale": "Applicable supporting view."} for i in evidence_ids]}


@pytest.mark.asyncio
async def test_unknown_links_fail_without_mutating_report(tmp_path):
    state = bound_state()
    old = state.structured_report.to_dict()
    result = await BuildFindingTool().call({"text": "Invented finding", "anatomy": "lung", "image_evidence_ids": ["MISSING"]}, context(state, tmp_path))
    assert result.is_error
    assert state.structured_report.to_dict() == old
    result = await LinkSupportTool().call({"target_id": "F001", "retrieved_evidence_ids": ["MISSING"]}, context(state, tmp_path))
    assert result.is_error
    assert state.structured_report.to_dict() == old


@pytest.mark.asyncio
async def test_impression_rejects_unknown_findings(tmp_path):
    state = bound_state()
    result = await BuildImpressionTool().call({"text": "Diagnosis", "linked_finding_ids": ["F999"]}, context(state, tmp_path))
    assert result.is_error
    assert len(state.structured_report.impressions) == 1


def test_evidence_requirements_follow_claim_type():
    state = bound_state()
    assert integrity_errors(state) == []  # Observations do not require unrelated literature.
    state.structured_report.findings[1].support.image_evidence = []
    assert any(identifier == "F002" and "visual" in description for identifier, _, description in integrity_errors(state))
    state.structured_report.findings[0].claim_type = "comparison"
    assert any("prior examination" in description for _, _, description in integrity_errors(state))
    state.structured_report.impressions[0].claim_type = "recommendation"
    assert any("medical knowledge" in description for _, _, description in integrity_errors(state))


@pytest.mark.asyncio
async def test_pass_requires_explicit_complete_claim_reviews(tmp_path):
    state = bound_state()
    _update_pipeline_state(state, AgentRole.QUALITY_REVIEW, "PASS, no issues")
    assert not state.review_passed
    state.review_issues.clear()
    tool = RecordClaimReviewTool()
    result = await tool.call({"reviews": [review("F001", ["IMG_A"]), review("F002", ["IMG_B"]), review("I001", ["IMG_A"])]}, context(state, tmp_path))
    assert not result.is_error
    _finalize_review(state)
    assert state.review_passed
    assert state.review_status == "PASS"


@pytest.mark.asyncio
async def test_relation_review_cannot_omit_or_invent_evidence(tmp_path):
    state = bound_state()
    result = await RecordClaimReviewTool().call({"reviews": [review("I001", [])]}, context(state, tmp_path))
    assert result.is_error
    assert not state.reviewed_claim_ids
    result = await RecordClaimReviewTool().call({"reviews": [review("I001", ["NOPE"])]}, context(state, tmp_path))
    assert result.is_error


@pytest.mark.asyncio
async def test_contradiction_blocks_pass_even_with_complete_coverage(tmp_path):
    state = bound_state()
    records = [review("F001", ["IMG_A"]), review("F002", ["IMG_B"]), review("I001", ["IMG_A"])]
    records[0]["evidence_relations"][0]["relation"] = "contradicted"
    await RecordClaimReviewTool().call({"reviews": records}, context(state, tmp_path))
    _finalize_review(state)
    assert state.review_completed and not state.review_passed


@pytest.mark.asyncio
async def test_local_repair_preserves_unrelated_claim_and_links(tmp_path):
    state = bound_state()
    unrelated = copy.deepcopy(state.structured_report.findings[1].to_dict())
    state.review_issues = [ReviewIssue("RI1", "certainty", "Qualify the opacity", attribution="wording", related_claim_id="F001")]
    state.iteration_count = 2
    _prepare_remediation(state)
    assert state.affected_node_ids == {"F001", "I001"}
    assert "Qualify the opacity" in _build_rerun_query(AgentRole.REPORT_WRITING, state)
    result = await BuildFindingTool().call({"finding_id": "F001", "text": "Possible right lung opacity.", "anatomy": "right lung", "image_evidence_ids": ["IMG_A"], "certainty": "possible"}, context(state, tmp_path))
    assert not result.is_error
    assert state.structured_report.findings[1].to_dict() == unrelated
    assert "I001" in state.dependency_graph.invalidated
    result = await BuildFindingTool().call({"finding_id": "F002", "text": "Changed", "anatomy": "pleura"}, context(state, tmp_path))
    assert result.is_error
    _finalize_review(state)
    assert any(i.related_claim_id == "I001" and i.issue_type == "traceability_failure" for i in state.review_issues)


@pytest.mark.asyncio
async def test_repair_replaces_old_dependencies(tmp_path):
    state = bound_state()
    state.iteration_count = 2
    state.review_issues = [ReviewIssue("RI1", "wrong_evidence", "Change view", related_claim_id="F001")]
    _prepare_remediation(state)
    result = await BuildFindingTool().call({"finding_id": "F001", "text": "No left pleural effusion.", "anatomy": "left pleura", "image_evidence_ids": ["IMG_B"]}, context(state, tmp_path))
    assert not result.is_error
    assert ("IMG_A", "F001") not in state.dependency_graph.edges
    assert ("IMG_B", "F001") in state.dependency_graph.edges


def test_renderer_keeps_exact_claim_spans_and_discards_agent_prose():
    state = bound_state()
    _update_pipeline_state(state, AgentRole.REPORT_WRITING, "FINDINGS: fabricated cancer")
    assert "fabricated" not in state.natural_language_report
    text = render_claims(state)
    assert len(state.rendered_claim_spans) == 3
    for span in state.rendered_claim_spans:
        assert text[span["start"]:span["end"]] == span["text"]


@pytest.mark.asyncio
async def test_pipeline_limit_keeps_revise_and_review_receives_views(monkeypatch, tmp_path):
    import src.core.orchestrator as module
    captured = []
    def builder(role, settings, state):
        return SimpleNamespace(role=role, system_prompt=role.value, max_turns=2, max_tokens=123,
                               temperature=0.7, hook_engine=None, tool_registry=ToolRegistry())
    async def loop(params):
        state = params._extra_context["pipeline_state"]
        role = params._extra_context["agent_role"]
        captured.append((role, params.max_tokens, params.images))
        if role == "visual_analysis" and not state.image_evidence:
            store = params._extra_context["image_store"]
            identifier = store.save(Image.new("RGB", (4, 4), "white"), "crop", "crop_roi")
            state.image_evidence.append(ImageEvidence(identifier, "crop"))
        if role == "report_writing" and state.structured_report is None:
            state.structured_report = StructuredReport(findings=[Finding("F001", "Opacity", support=SupportEvidence(image_evidence=state.image_evidence))], impressions=[Impression("I001", "Opacity", linked_findings=["F001"])])
            for claim in state.structured_report.findings + state.structured_report.impressions:
                sync_claim_graph(state, claim)
        yield StreamEvent(EventType.DONE, {"reason": "complete", "final_text": "PASS no issues"})
    monkeypatch.setattr(module, "build_agent", builder)
    monkeypatch.setattr(module, "query_loop", loop)
    bundle = SimpleNamespace(frozen_manifest=None, versions={"harness": "h0"}, stage_instructions=lambda *args: "")
    events = [e async for e in run_pipeline(Image.new("RGB", (8, 8)), {}, {"data": {"output_dir": str(tmp_path)}, "pipeline": {"max_iterations": 3}}, object(), runtime_bundle=bundle)]
    result = events[-1].data
    assert result["iterations"] == 3
    assert result["review_status"] == "REVISE"
    assert result["review_passed"] is False and result["remaining_issues"]
    assert all(tokens == 123 for _, tokens, _ in captured)
    assert all(len(images) >= 2 for role, _, images in captured if role == "quality_review")
    assert (tmp_path / "records.sqlite").exists()

@pytest.mark.asyncio
async def test_review_cannot_substitute_for_recomputation(tmp_path):
    state = bound_state()
    state.iteration_count = 2
    state.review_issues = [ReviewIssue("RI1", "wrong_evidence", "Recompute affected claim", related_claim_id="F001")]
    _prepare_remediation(state)
    await RecordClaimReviewTool().call({"reviews": [review("F001", ["IMG_A"]), review("F002", ["IMG_B"]), review("I001", ["IMG_A"])]}, context(state, tmp_path))
    _finalize_review(state)
    assert not state.review_passed
    assert {"F001", "I001"} <= state.dependency_graph.invalidated


@pytest.mark.asyncio
async def test_binding_cannot_refresh_stale_upstream_evidence(tmp_path):
    state = bound_state()
    state.iteration_count = 2
    state.review_issues = [ReviewIssue("RI1", "visual_error", "Reinspect image", attribution="visual", related_claim_id="F001")]
    _prepare_remediation(state)
    assert "IMG_A" in state.dependency_graph.invalidated
    result = await BuildFindingTool().call({"finding_id": "F001", "text": "Opacity", "anatomy": "lung", "image_evidence_ids": ["IMG_A"]}, context(state, tmp_path))
    assert result.is_error
    assert "IMG_A" in state.dependency_graph.invalidated


@pytest.mark.asyncio
async def test_resume_uses_checkpoint_without_rerunning_upstream(monkeypatch, tmp_path):
    import src.core.orchestrator as module
    from src.data.image_store import SessionImageStore
    captured = []
    state = bound_state()
    state.original_image = Image.new("RGB", (8, 8))
    store = SessionImageStore("resume")
    store.save(state.original_image, "Original", "input")
    bundle = SimpleNamespace(frozen_manifest=None, versions={"harness": "h1"}, stage_instructions=lambda *args: "fixed runtime")
    def builder(role, settings, state):
        return SimpleNamespace(role=role, system_prompt=role.value, max_turns=1, max_tokens=100,
                               temperature=0.7, hook_engine=None, tool_registry=ToolRegistry())
    async def loop(params):
        captured.append(params._extra_context["agent_role"])
        yield StreamEvent(EventType.DONE, {"reason": "complete", "final_text": "done"})
    checkpoints = []
    async def checkpoint(stage, current, images):
        checkpoints.append((stage, current.iteration_count, images))
    monkeypatch.setattr(module, "build_agent", builder)
    monkeypatch.setattr(module, "query_loop", loop)
    events = [e async for e in run_pipeline(state.original_image, {}, {"data": {"output_dir": str(tmp_path)}, "pipeline": {"max_iterations": 1}}, object(), initial_state=state, image_store=store, start_stage="report_writing", runtime_bundle=bundle, checkpoint_callback=checkpoint)]
    assert captured == ["report_writing", "quality_review"]
    assert [stage for stage, _, _ in checkpoints] == captured
    assert all(iteration == 1 for _, iteration, _ in checkpoints)
    assert events[-1].data["versions"] == {"harness": "h1"}

@pytest.mark.asyncio
async def test_shared_view_selects_binding_for_the_requested_observation(tmp_path):
    state = PipelineState(session_id="shared", iteration_count=1)
    state.observation_graph = [{"observation_id": "OBS_LEFT"}, {"observation_id": "OBS_RIGHT"}]
    state.image_evidence = [
        ImageEvidence("IMG_SHARED", "Full study", finding_description="No left effusion", observation_id="OBS_LEFT"),
        ImageEvidence("IMG_SHARED", "Full study", finding_description="Right opacity", observation_id="OBS_RIGHT"),
    ]
    result = await BuildFindingTool().call({"text": "No left effusion", "anatomy": "left pleura", "image_evidence_ids": ["IMG_SHARED"], "observation_ids": ["OBS_LEFT"]}, context(state, tmp_path))
    assert not result.is_error
    support = state.structured_report.findings[0].support.image_evidence
    assert len(support) == 1 and support[0].observation_id == "OBS_LEFT"
    assert ("OBS_LEFT", "F001") in state.dependency_graph.edges
    assert ("OBS_RIGHT", "F001") not in state.dependency_graph.edges
