"""Explicit clinical review records; deterministic completion never infers PASS from prose."""
from src.core.tool import Tool
from src.core.types import ToolResult
from src.report.grounding import RELATIONS, claim_id, claims, effective_support, get_claim, register_issue

CHECKS = ("visual_support", "evidence_applicability", "findings_impression_consistency", "reporting_conventions")


class RecordClaimReviewTool(Tool):
    name = "record_claim_review"
    description = "Record explicit per-claim review of all four dimensions and evidence applicability; record observation omission coverage"
    is_read_only = False
    input_schema = {"type": "object", "properties": {
        "reviews": {"type": "array", "items": {"type": "object", "properties": {
            "claim_id": {"type": "string"},
            **{name: {"type": "boolean"} for name in CHECKS},
            "rationale": {"type": "string"},
            "evidence_relations": {"type": "array", "items": {"type": "object", "properties": {
                "evidence_id": {"type": "string"}, "relation": {"type": "string", "enum": list(RELATIONS)},
                "rationale": {"type": "string"},
            }, "required": ["evidence_id", "relation", "rationale"]}},
        }, "required": ["claim_id", *CHECKS, "rationale", "evidence_relations"]}},
        "reviewed_observation_ids": {"type": "array", "items": {"type": "string"}, "description": "Observed nodes explicitly compared against the report for omissions; attribute omissions separately."},
    }, "required": ["reviews"]}

    def prompt(self):
        return ("Review EVERY finding and impression against current images, applicable evidence and supporting findings. "
                "For each record all four checks, rationale, and support relation for every linked evidence ID, including "
                "inherited impression evidence. Relatedness is not full support. Record observation IDs checked for omissions "
                "and attribute each omission separately. Missing review records prevent PASS; model prose cannot override this gate.")

    async def call(self, args, context):
        state = context.extra.get("pipeline_state")
        if state is None or state.structured_report is None:
            return ToolResult("Error: no report to review", is_error=True)
        pending = []
        for review in args.get("reviews", []):
            target = get_claim(state, review.get("claim_id", ""))
            if target is None or any(type(review.get(name)) is not bool for name in CHECKS) or not review.get("rationale", "").strip():
                return ToolResult("Error: each review requires an existing claim, four boolean checks and rationale", is_error=True)
            support = effective_support(state, target)
            expected = {e.image_id for e in support.image_evidence} | {e.evidence_id for e in support.retrieved_evidence} | set(support.prior_examination_ids)
            relations = review.get("evidence_relations", [])
            received = {r.get("evidence_id") for r in relations}
            if expected != received or len(received) != len(relations) or any(r.get("relation") not in RELATIONS or not r.get("rationale", "").strip() for r in relations):
                return ToolResult(f"Error: review {claim_id(target)} must assess exactly its linked evidence IDs: {sorted(expected)}", is_error=True)
            pending.append((target, review))
        observations = {o.get("observation_id") for o in state.observation_graph if o.get("observation_id")}
        seen = set(args.get("reviewed_observation_ids", []))
        if seen - observations:
            return ToolResult("Error: unknown reviewed observation IDs", is_error=True)
        for target, review in pending:
            identifier = claim_id(target)
            state.claim_reviews[identifier] = dict(review)
            state.reviewed_claim_ids.add(identifier)
            target.support.relations = list(review["evidence_relations"])
            failed = [name for name in CHECKS if not review[name]]
            if failed:
                category = "internal_inconsistency" if "findings_impression_consistency" in failed else "evidence_mismatch"
                register_issue(state, identifier, category, f"Failed review dimensions {failed}: {review['rationale']}")
            for relation in review["evidence_relations"]:
                if relation["relation"] in {"contradicted", "irrelevant", "unverifiable"}:
                    register_issue(state, identifier, "evidence_mismatch", f"{relation['evidence_id']} {relation['relation']}: {relation['rationale']}")
        state.reviewed_observation_ids.update(seen)
        state.review_completed = ({claim_id(c) for c in claims(state)} <= state.reviewed_claim_ids and observations <= state.reviewed_observation_ids)
        return ToolResult(f"Recorded {len(pending)} claim reviews; complete coverage={state.review_completed}")
