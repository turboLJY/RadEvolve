"""
StandardizeObservationTool — maps free-text observations to structured
RadLex / SNOMED terms.  Prefers FAISS vector search when available,
falls back to fuzzy matching against local JSON knowledge bases.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult, ValidationResult
from src.kb.abbreviation_expander import AbbreviationExpander
from src.tools.evidence.provenance import stable_id, prepare_record_update

if TYPE_CHECKING:
    from src.kb.base import KnowledgeSource

logger = logging.getLogger(__name__)
_expander = AbbreviationExpander()


class StandardizeObservationTool(Tool):
    name = "standardize_observation"
    description = (
        "Map a free-text radiologic observation to standardized RadLex and "
        "SNOMED CT terms.  Returns the best-matching standardized term, codes, "
        "synonyms, and definition.  Stores the result in the observation graph."
    )
    input_schema: dict = {
        "type": "object",
        "properties": {
            "observation_id": {"type": "string", "description": "Existing visual observation ID to normalize; preserve it when provided."},
            "anatomy": {"type": "string"},
            "laterality": {"type": "string", "enum": ["left", "right", "bilateral", "midline", "unspecified"]},
            "polarity": {"type": "string", "enum": ["positive", "negative", "uncertain"]},
            "certainty": {"type": "string"},
            "attributes": {"type": "object"},
            "image_evidence_ids": {"type": "array", "items": {"type": "string"}},
            "observation_text": {
                "type": "string",
                "description": "Free-text observation to standardize (e.g. 'fluid around the lungs').",
            }
        },
        "required": ["observation_text"],
        "additionalProperties": False,
    }
    is_read_only = False
    is_concurrency_safe = False

    def __init__(
        self,
        knowledge_dir: str = "./data/knowledge",
        radlex_source: KnowledgeSource | None = None,
        snomed_source: KnowledgeSource | None = None,
    ) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._radlex_source = radlex_source
        self._snomed_source = snomed_source

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------

    def prompt(self) -> str:
        return (
            "Use standardize_observation to map a free-text radiologic observation "
            "(e.g. 'bilateral haziness at the bases') to its standardized RadLex and "
            "SNOMED CT codes.  This is typically the first step when processing each "
            "observation from the visual analysis stage."
        )

    def validate_input(self, args: dict) -> ValidationResult:
        text = args.get("observation_text")
        if not text or not isinstance(text, str) or not text.strip():
            return ValidationResult(valid=False, message="observation_text must be a non-empty string")
        return ValidationResult(valid=True)

    async def _vector_radlex_match(self, text: str) -> tuple[dict | None, float]:
        try:
            if self._radlex_source and self._radlex_source.is_available():
                hits = await self._radlex_source.search(text, max_results=1)
                if hits:
                    h = hits[0]
                    return {
                        "preferred_name": h.preferred_term,
                        "code": h.source_id,
                        "synonyms": h.metadata.get("synonyms", []),
                        "definition": h.content,
                        "category": h.metadata.get("category", ""),
                    }, h.score
        except Exception as e:
            if getattr(self._radlex_source, "snapshot", None) is not None:
                raise
            logger.warning("RadLex vector search failed: %s", e)
        return None, 0.0

    async def _vector_snomed_match(self, text: str) -> tuple[dict | None, float]:
        try:
            if self._snomed_source and self._snomed_source.is_available():
                hits = await self._snomed_source.search(text, max_results=1)
                if hits:
                    h = hits[0]
                    return {
                        "sctid": h.source_id,
                        "term": h.preferred_term,
                        "synonyms": h.metadata.get("synonyms", []),
                        "definition": h.content,
                    }, h.score
        except Exception as e:
            if getattr(self._snomed_source, "snapshot", None) is not None:
                raise
            logger.warning("SNOMED vector search failed: %s", e)
        return None, 0.0

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        observation_text: str = _expander.expand_query(args["observation_text"].strip())

        radlex_match, radlex_score = await self._vector_radlex_match(observation_text)

        snomed_match, snomed_score = await self._vector_snomed_match(observation_text)

        result: dict[str, Any] = {
            "input_text": observation_text,
            "standardized_term": None,
            "radlex_code": None,
            "snomed_code": None,
            "synonyms": [],
            "definition": None,
            "match_scores": {"radlex": radlex_score, "snomed": snomed_score},
        }

        if radlex_match:
            result["standardized_term"] = radlex_match["preferred_name"]
            result["radlex_code"] = radlex_match["code"]
            result["synonyms"] = radlex_match.get("synonyms", [])
            result["definition"] = radlex_match.get("definition")

        if snomed_match:
            result["snomed_code"] = snomed_match["sctid"]
            if result["standardized_term"] is None:
                result["standardized_term"] = snomed_match["term"]
                result["synonyms"] = snomed_match.get("synonyms", [])
                result["definition"] = snomed_match.get("definition")

        # Normalization must not discard negation, uncertainty, laterality, or source images.
        original_text = args["observation_text"].strip()
        lower = original_text.lower()
        uncertain = bool(re.search(r"\b(possible|possibly|may|might|cannot exclude|suspected|questionable)\b", lower))
        negated = bool(re.search(r"\b(no|without|absent|negative for)\b", lower)) and not uncertain
        laterality = "bilateral" if "bilateral" in lower else "left" if re.search(r"\bleft\b", lower) else "right" if re.search(r"\bright\b", lower) else "unspecified"
        pipeline_state = context.extra.get("pipeline_state")
        previous = next((o for o in pipeline_state.observation_graph if o.get("observation_id") == args.get("observation_id")), None) if pipeline_state else None
        if args.get("observation_id") and previous is None:
            return ToolResult(data=json.dumps({"error": "Unknown observation_id"}), is_error=True)
        image_ids = args.get("image_evidence_ids", (previous or {}).get("image_evidence_ids", []))
        if pipeline_state is not None:
            known_images = {e.image_id for e in pipeline_state.image_evidence}
            if any(i not in known_images for i in image_ids):
                return ToolResult(data=json.dumps({"error": "Unknown visual evidence ID"}), is_error=True)
        features = {
            "anatomy": args.get("anatomy", (previous or {}).get("anatomy", "")),
            "laterality": args.get("laterality", (previous or {}).get("laterality", laterality)),
            "polarity": args.get("polarity", (previous or {}).get("polarity", "uncertain" if uncertain else "negative" if negated else "positive")),
            "certainty": args.get("certainty", (previous or {}).get("certainty", "uncertain" if uncertain else "stated")),
            "attributes": args.get("attributes", (previous or {}).get("attributes", {})),
            "image_evidence_ids": image_ids,
        }
        result.update(features)
        result["observation_id"] = args.get("observation_id") or stable_id("OBS", [original_text, features])
        result["original_text"] = original_text
        if pipeline_state is not None:
            obs_node = {
                "observation_id": result["observation_id"],
                **features,
                "original_text": original_text,
                "standardized_term": result["standardized_term"],
                "radlex_code": result["radlex_code"],
                "snomed_code": result["snomed_code"],
                "synonyms": result["synonyms"],
                "definition": result["definition"],
            }
            try:
                prepare_record_update(pipeline_state, obs_node["observation_id"], "observation", obs_node)
            except ValueError as exc:
                return ToolResult(data=json.dumps({"error": str(exc)}), is_error=True)
            if previous:
                pipeline_state.observation_graph[pipeline_state.observation_graph.index(previous)] = obs_node
            elif not any(o.get("observation_id") == obs_node["observation_id"] for o in pipeline_state.observation_graph):
                pipeline_state.observation_graph.append(obs_node)
            graph = getattr(pipeline_state, "dependency_graph", None)
            if graph is not None:
                graph.add_node(obs_node["observation_id"], "observation", obs_node)
                for image_id in image_ids:
                    if image_id not in graph.nodes:
                        graph.add_node(image_id, "image", {"image_id": image_id})
                    graph.add_edge(image_id, obs_node["observation_id"])
                graph.set_dependencies(obs_node["observation_id"], image_ids)

        return ToolResult(data=json.dumps(result, indent=2, ensure_ascii=False),
                          metadata={"produced_ids": [result["observation_id"]], "consumed_ids": image_ids})
