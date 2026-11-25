"""
CaseRetrievalTool — retrieves similar cases via remote HTTP service.
Sends joined observations as a dense query to POST /search/cases.
"""

from __future__ import annotations

import json
import logging

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import RetrievedEvidence, ToolResult, ValidationResult
from src.tools.evidence.provenance import add_evidence, observation_ids, stable_id

logger = logging.getLogger(__name__)

_MODALITY_TO_CATEGORY: dict[str, str] = {
    "chest_xray": "chest_xray",
    "cxr": "chest_xray",
    "chest_x-ray": "chest_xray",
    "chest_ct": "chest_ct",
    "brain_ct": "brain_ct",
    "head_ct": "brain_ct",
    "brain_mri": "brain_mri",
    "abdominal_ct": "abdominal_ct",
    "abdomen_ct": "abdominal_ct",
}
_DEFAULT_CATEGORY = "chest_xray"


class CaseRetrievalTool(Tool):
    name = "case_retrieval"
    description = (
        "Find similar cases whose findings best match a given set of observations. "
        "Returns case reports with findings, impressions, and structured findings lists."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "observation_ids": {"type": "array", "items": {"type": "string"}, "description": "Observation IDs motivating retrieval."},
            "observations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of observation terms to match against case reports.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of cases to return.",
                "default": 5,
                "minimum": 1,
                "maximum": 50,
            },
        },
        "required": ["observations"],
        "additionalProperties": False,
    }
    is_read_only = False
    is_concurrency_safe = False

    def __init__(
        self,
        remote_base_url: str | None = None,
        timeout: float = 10.0,
        snapshot=None,
    ) -> None:
        self._base_url = remote_base_url.rstrip("/") if remote_base_url else None
        self._timeout = timeout
        self._evidence_counter = 0
        self._snapshot = snapshot

    def _next_evidence_id(self) -> str:
        self._evidence_counter += 1
        return f"CASE_{self._evidence_counter:03d}"

    def prompt(self) -> str:
        return (
            "Use case_retrieval to find similar cases whose findings overlap "
            "with the current observations. Helps correlate the current study "
            "with known diagnostic patterns."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        obs = args.get("observations")
        if not obs or not isinstance(obs, list) or len(obs) == 0:
            return ValidationResult(valid=False, message="observations must be a non-empty list of strings")
        if not all(isinstance(o, str) for o in obs):
            return ValidationResult(valid=False, message="all observation items must be strings")
        max_r = args.get("max_results", 5)
        if not isinstance(max_r, int) or max_r < 1:
            return ValidationResult(valid=False, message="max_results must be a positive integer")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        observations = args["observations"]
        max_results = args.get("max_results", 5)
        state = context.extra.get("pipeline_state")
        category = _MODALITY_TO_CATEGORY.get((state.modality if state else "").lower(), _DEFAULT_CATEGORY)
        query = " ".join(observations)
        try:
            inputs = observation_ids(args, state)
            if self._snapshot is not None:
                hits = self._snapshot.search(query, ["case"], category, max_results)
            elif self._base_url:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(f"{self._base_url}/search/cases", json={"query": query, "category": category, "max_results": max_results})
                    response.raise_for_status()
                    hits = response.json().get("results", [])
            else:
                hits = []
            results = []
            for hit in hits:
                sections = hit.get("sections", {})
                content = hit.get("content") or f"Findings: {sections.get('findings', '')} Impression: {sections.get('impression', '')}"
                source_id = hit.get("document_id") or hit.get("case_id", "")
                eid = stable_id("CASE", [hit.get("snapshot_version", "live"), source_id, content])
                evidence = RetrievedEvidence(eid, "case", content, float(hit.get("score", 0)), metadata={
                    **hit.get("metadata", {}), "case_id": source_id, "query": query,
                    "snapshot_version": hit.get("snapshot_version", ""), "support_relation": "unassessed",
                    "patient_specific_evidence": False,
                })
                add_evidence(state, evidence, inputs)
                results.append(evidence.to_dict())
            return ToolResult(data=json.dumps({"observations_query": observations, "category": category,
                                              "total_matches": len(results), "results": results}, ensure_ascii=False),
                              metadata={"produced_ids": [r["evidence_id"] for r in results], "consumed_ids": inputs, "evidence_ids": [r["evidence_id"] for r in results], "observation_ids": inputs})
        except (ValueError, OSError, httpx.HTTPError) as exc:
            return ToolResult(data=json.dumps({"error": f"Case retrieval failed: {type(exc).__name__}: {exc}"}), is_error=True)
