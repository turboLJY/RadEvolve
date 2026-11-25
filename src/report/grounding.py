"""Claim binding, dependency maintenance and deterministic report integrity checks.

These checks enforce traceability, not clinical truth. Clinical support relations
are supplied explicitly by the review agent and remain visible in the record.
"""
from __future__ import annotations

from src.core.types import Finding, Impression, PipelineState, ReviewIssue, SupportEvidence

CLAIM_TYPES = ("observation", "normality", "attribute", "comparison", "diagnostic", "recommendation")
POLARITIES = ("positive", "negative", "uncertain", "unspecified")
RELATIONS = ("fully_supported", "partially_supported", "contradicted", "irrelevant", "unverifiable")
CLAIM_PROPERTIES = {
    "claim_type": {"type": "string", "enum": list(CLAIM_TYPES)},
    "anatomy": {"type": "string"},
    "laterality": {"type": "string"},
    "polarity": {"type": "string", "enum": list(POLARITIES)},
    "attributes": {"type": "object", "description": "Material size, morphology, severity, signal or other attributes."},
    "certainty": {"type": "string", "description": "Certainty expressed in the claim, preserved during rendering."},
    "observation_ids": {"type": "array", "items": {"type": "string"}},
    "prior_examination_ids": {"type": "array", "items": {"type": "string"}},
}


def claim_id(claim) -> str:
    return claim.finding_id if isinstance(claim, Finding) else claim.impression_id


def claims(state: PipelineState) -> list:
    report = state.structured_report
    return list(report.findings) + list(report.impressions) if report else []


def get_claim(state: PipelineState, identifier: str):
    return next((c for c in claims(state) if claim_id(c) == identifier), None)


def prior_ids(state: PipelineState) -> set[str]:
    """Only IDs of actual prior examinations supplied in the reporting input."""
    priors = state.exam_info.get("prior_examinations", [])
    return {p["examination_id"] for p in priors if isinstance(p, dict) and p.get("examination_id") and (p.get("images") or p.get("report") or p.get("image_ids"))}


def resolve_support(state: PipelineState, args: dict) -> SupportEvidence:
    image_ids = list(dict.fromkeys(args.get("image_evidence_ids", [])))
    evidence_ids = list(dict.fromkeys(args.get("retrieved_evidence_ids", [])))
    images = {e.image_id: e for e in state.image_evidence}
    evidence = {e.evidence_id: e for e in state.retrieved_evidence}
    unknown = set(image_ids) - images.keys() | (set(evidence_ids) - evidence.keys())
    if unknown:
        raise ValueError(f"Unknown evidence IDs: {sorted(unknown)}")
    stale = (set(image_ids) | set(evidence_ids) | set(args.get("observation_ids", []))) & state.dependency_graph.invalidated
    if stale:
        raise ValueError(f"Affected evidence must be recomputed before claim binding: {sorted(stale)}")
    previous_ids = list(dict.fromkeys(args.get("prior_examination_ids", [])))
    if set(previous_ids) - prior_ids(state):
        raise ValueError("Comparison evidence must identify prior examinations actually supplied in the input")
    observations = {o.get("observation_id") for o in state.observation_graph}
    unknown_obs = set(args.get("observation_ids", [])) - observations
    if unknown_obs:
        raise ValueError(f"Unknown observation IDs: {sorted(unknown_obs)}")
    selected_observations = set(args.get("observation_ids", []))
    selected_images = [e for e in state.image_evidence if e.image_id in image_ids
                       and (not selected_observations or not e.observation_id or e.observation_id in selected_observations)]
    if set(image_ids) - {e.image_id for e in selected_images}:
        raise ValueError("Requested image bindings do not belong to the selected observations")
    return SupportEvidence(image_evidence=selected_images,
                           retrieved_evidence=[evidence[i] for i in evidence_ids],
                           prior_examination_ids=previous_ids)


def effective_support(state: PipelineState, claim) -> SupportEvidence:
    support = SupportEvidence(image_evidence=list(claim.support.image_evidence),
                              retrieved_evidence=list(claim.support.retrieved_evidence),
                              prior_examination_ids=list(claim.support.prior_examination_ids),
                              relations=list(claim.support.relations))
    if isinstance(claim, Impression) and state.structured_report:
        for finding in state.structured_report.findings:
            if finding.finding_id in claim.linked_findings:
                support.image_evidence.extend(finding.support.image_evidence)
                support.retrieved_evidence.extend(finding.support.retrieved_evidence)
                support.prior_examination_ids.extend(finding.support.prior_examination_ids)
    return support


def ensure_mutable(state: PipelineState, identifier: str) -> None:
    if state.iteration_count > 1 and identifier not in state.affected_node_ids:
        raise ValueError(f"Claim {identifier} is outside the dependency repair set and must be preserved")


def sync_claim_graph(state: PipelineState, claim) -> None:
    graph = state.dependency_graph
    identifier = claim_id(claim)
    graph.add_node(identifier, "finding" if isinstance(claim, Finding) else "impression", claim.to_dict())
    dependencies = list(claim.observation_ids)
    for ev in claim.support.image_evidence:
        if ev.image_id not in graph.nodes:
            graph.add_node(ev.image_id, "image_evidence", ev.to_dict())
        dependencies.append(ev.image_id)
    for ev in claim.support.retrieved_evidence:
        if ev.evidence_id not in graph.nodes:
            graph.add_node(ev.evidence_id, "retrieved_evidence", ev.to_dict())
        dependencies.append(ev.evidence_id)
    dependencies.extend(claim.support.prior_examination_ids)
    if isinstance(claim, Impression):
        dependencies.extend(claim.linked_findings)
    for parent in dependencies:
        # Observation/prior nodes are registered when acquired; missing graph
        # nodes are still explicit records, never inferred support relations.
        if parent in claim.support.prior_examination_ids:
            graph.add_node(parent, "prior_examination", {"examination_id": parent})
    for source in dependencies:
        if source not in graph.nodes:
            observation = next((o for o in state.observation_graph if o.get("observation_id") == source), None)
            supporting = get_claim(state, source)
            if observation is not None:
                graph.add_node(source, "observation", observation)
            elif supporting is not None:
                graph.add_node(source, "finding", supporting.to_dict())
    graph.set_dependencies(identifier, dependencies)
    if state.iteration_count > 1:
        state.affected_node_ids.update({identifier} | graph.descendants({identifier}))
        state.reviewed_claim_ids.difference_update(state.affected_node_ids)
        for affected in state.affected_node_ids:
            state.claim_reviews.pop(affected, None)
        state.repaired_node_ids.add(identifier)
    state.natural_language_report = ""
    state.rendered_claim_spans = []
    state.review_completed = False
    state.review_passed = False


def register_issue(state: PipelineState, target: str, category: str, description: str,
                   severity: str = "high", attribution: str = "generation") -> ReviewIssue:
    from src.core.types import AgentRole
    for issue in state.review_issues:
        if (issue.related_claim_id, issue.issue_type, issue.description) == (target, category, description):
            return issue
    role = {"visual": AgentRole.VISUAL_ANALYSIS, "retrieval": AgentRole.EVIDENCE_RETRIEVAL}.get(attribution, AgentRole.REPORT_WRITING)
    issue = ReviewIssue(issue_id=f"RI{state.iteration_count:02d}_{len(state.review_issues)+1:04d}",
                        issue_type=category, description=description, severity=severity,
                        attribution=attribution, target_agent=role, related_claim_id=target,
                        evidence_mismatch=description, suggested_fix="Correct the affected claim and preserve unaffected claims.")
    state.review_issues.append(issue)
    return issue


def integrity_errors(state: PipelineState) -> list[tuple[str, str, str]]:
    errors = []
    report = state.structured_report
    if not report or not report.findings or not report.impressions:
        errors.append(("", "report_structure", "Report requires findings and impressions"))
        if not report:
            return errors
    image_ids = {e.image_id for e in state.image_evidence}
    evidence_ids = {e.evidence_id for e in state.retrieved_evidence}
    finding_ids = {f.finding_id for f in report.findings}
    all_ids = [claim_id(c) for c in claims(state)]
    if len(all_ids) != len(set(all_ids)):
        errors.append(("", "traceability_failure", "Duplicate claim identifiers"))
    for claim in claims(state):
        identifier = claim_id(claim)
        support = effective_support(state, claim)
        if any(e.image_id not in image_ids for e in support.image_evidence) or any(e.evidence_id not in evidence_ids for e in support.retrieved_evidence):
            errors.append((identifier, "traceability_failure", "Claim references unavailable evidence"))
        if claim.claim_type not in CLAIM_TYPES:
            errors.append((identifier, "traceability_failure", "Unknown claim type"))
        if claim.claim_type in {"observation", "normality", "attribute", "comparison", "diagnostic"} and not support.image_evidence:
            errors.append((identifier, "traceability_failure", "Patient-specific claim requires current-examination visual evidence"))
        if claim.claim_type == "comparison" and not (set(support.prior_examination_ids) & prior_ids(state)):
            errors.append((identifier, "traceability_failure", "Temporal comparison lacks an available prior examination"))
        if claim.claim_type == "recommendation" and not any(e.source_type != "case" for e in support.retrieved_evidence):
            errors.append((identifier, "traceability_failure", "Recommendation requires applicable medical knowledge"))
        if isinstance(claim, Impression) and (not claim.linked_findings or set(claim.linked_findings) - finding_ids):
            errors.append((identifier, "internal_inconsistency", "Impression requires valid supporting finding IDs"))
    return errors


def render_claims(state: PipelineState) -> str:
    """Render exact structured claim text and character ranges (end exclusive)."""
    report = state.structured_report
    if report is None:
        return ""
    text = "FINDINGS:\n"
    spans = []
    for index, finding in enumerate(report.findings):
        if index:
            text += " "
        start = len(text)
        text += finding.text
        spans.append({"claim_id": finding.finding_id, "section": "FINDINGS", "start": start, "end": len(text), "text": finding.text})
    text += "\n\nIMPRESSION:\n"
    for index, impression in enumerate(report.impressions, 1):
        if index > 1:
            text += "\n"
        text += f"{index}. "
        start = len(text)
        text += impression.text
        spans.append({"claim_id": impression.impression_id, "section": "IMPRESSION", "start": start, "end": len(text), "text": impression.text})
    state.natural_language_report = text
    state.rendered_claim_spans = spans
    report.metadata["claim_spans"] = spans
    return text
