"""
LiteratureSearchTool — dual-mode literature search:
1) Remote HTTP service for clinical guidelines (POST /search/guidelines)
2) PubMed E-utilities for online literature
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import httpx

from src.core.tool import Tool, ToolUseContext
from src.core.types import RetrievedEvidence, ToolResult, ValidationResult
from src.kb.abbreviation_expander import AbbreviationExpander
from src.tools.evidence.provenance import add_evidence, observation_ids, stable_id

if TYPE_CHECKING:
    from src.kb.pubmed_client import PubMedClient

logger = logging.getLogger(__name__)
_expander = AbbreviationExpander()

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


class LiteratureSearchTool(Tool):
    name = "literature_search"
    description = (
        "Search clinical guidelines and PubMed for evidence related to a query. "
        "Returns matched guidelines and articles ranked by relevance and stores "
        "them as RetrievedEvidence records."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "observation_ids": {"type": "array", "items": {"type": "string"}, "description": "Observation IDs motivating this query."},
            "query": {
                "type": "string",
                "description": "Natural-language query to search (e.g. 'management of solitary pulmonary nodule').",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 5,
                "minimum": 1,
                "maximum": 30,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    is_read_only = False
    is_concurrency_safe = False
    max_result_size_chars = 30000

    def __init__(
        self,
        remote_base_url: str | None = None,
        timeout: float = 10.0,
        pubmed_client: PubMedClient | None = None,
        snapshot=None,
    ) -> None:
        self._base_url = remote_base_url.rstrip("/") if remote_base_url else None
        self._timeout = timeout
        self._pubmed_client = pubmed_client
        self._evidence_counter = 0
        self._snapshot = snapshot

    def _next_evidence_id(self) -> str:
        self._evidence_counter += 1
        return f"LIT_{self._evidence_counter:03d}"

    def prompt(self) -> str:
        return (
            "Use literature_search to find clinical guidelines and PubMed articles "
            "relevant to a specific radiologic finding or clinical question."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        query = args.get("query")
        if not query or not isinstance(query, str) or not query.strip():
            return ValidationResult(valid=False, message="query must be a non-empty string")
        max_r = args.get("max_results", 5)
        if not isinstance(max_r, int) or max_r < 1:
            return ValidationResult(valid=False, message="max_results must be a positive integer")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        query = _expander.expand_query(args["query"].strip())
        max_results = args.get("max_results", 5)
        state = context.extra.get("pipeline_state")
        category = _MODALITY_TO_CATEGORY.get((state.modality if state else "").lower(), _DEFAULT_CATEGORY)
        try:
            inputs = observation_ids(args, state)
            hits = []
            warnings = []
            if self._snapshot is not None:
                sources = self._snapshot.sources & {"guideline", "pubmed", "literature"}
                if not sources:
                    raise ValueError("Frozen corpus does not declare a literature or guideline source")
                hits = self._snapshot.search(query, sources, category, max_results)
            else:
                async def guidelines():
                    if not self._base_url:
                        return []
                    async with httpx.AsyncClient(timeout=self._timeout) as client:
                        response = await client.post(f"{self._base_url}/search/guidelines", json={"query": query, "category": category, "max_results": max_results})
                        response.raise_for_status()
                        return [{**h, "source_type": "guideline", "content": h.get("text", ""),
                                 "document_id": h.get("chunk_id") or h.get("doc_id", ""), "retrieval_mode": "remote_guidelines"}
                                for h in response.json().get("results", [])]
                async def pubmed():
                    if self._pubmed_client is None:
                        return []
                    return [{"source_type": "pubmed", "content": h.content, "title": h.preferred_term,
                             "document_id": h.source_id, "score": h.score, "metadata": h.metadata, "retrieval_mode": "pubmed"}
                            for h in await self._pubmed_client.search(query, max_results)]
                for result in await asyncio.gather(guidelines(), pubmed(), return_exceptions=True):
                    if isinstance(result, BaseException):
                        warnings.append(f"Retrieval source failed: {type(result).__name__}")
                    else:
                        hits.extend(result)
            results = []
            for hit in hits:
                eid = stable_id("LIT", [hit.get("snapshot_version", "live"), hit.get("document_id"), hit.get("content", "")])
                metadata = {**hit.get("metadata", {}), "document_id": hit.get("document_id", ""),
                            "title": hit.get("title", ""), "snapshot_version": hit.get("snapshot_version", ""),
                            "query": query, "retrieval_mode": hit.get("retrieval_mode", ""),
                            "support_relation": "unassessed"}
                evidence = RetrievedEvidence(eid, hit["source_type"], hit.get("content", ""),
                                             float(hit.get("score", 0)), metadata=metadata)
                add_evidence(state, evidence, inputs)
                results.append(evidence.to_dict())
            return ToolResult(data=json.dumps({"query": query, "category": category, "total_matches": len(results),
                                              "results": results, "warnings": warnings}, ensure_ascii=False),
                              metadata={"produced_ids": [r["evidence_id"] for r in results], "consumed_ids": inputs, "evidence_ids": [r["evidence_id"] for r in results], "observation_ids": inputs})
        except (ValueError, OSError) as exc:
            return ToolResult(data=json.dumps({"error": str(exc)}), is_error=True)
