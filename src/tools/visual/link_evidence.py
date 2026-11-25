"""
LinkEvidenceTool — Link visual findings to supporting image evidence.

This is the most important tool in the visual analysis pipeline.  It
creates the ``ImageEvidence`` binding that ties a textual finding
description to one or more processed images, forming a traceable
evidence chain.  The quality-review agent later checks that every
reported finding has at least one linked evidence record.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
import re
from src.tools.evidence.provenance import stable_id, prepare_record_update

from src.core.tool import Tool, ToolUseContext
from src.core.types import ImageEvidence, PipelineState, ToolResult

logger = logging.getLogger(__name__)


class LinkEvidenceTool(Tool):
    name = "link_visual_evidence"
    description = (
        "Link visual findings to supporting image evidence, creating a traceable "
        "evidence chain. For each finding, specify the finding description, the "
        "image_ids that support it, and a confidence score. This is how visual "
        "observations become auditable, evidence-backed findings."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "observation_id": {"type": "string", "description": "Existing observation ID when revising a visual observation."},
            "anatomy": {"type": "string"},
            "laterality": {"type": "string", "enum": ["left", "right", "bilateral", "midline", "unspecified"]},
            "polarity": {"type": "string", "enum": ["positive", "negative", "uncertain"]},
            "certainty": {"type": "string"},
            "attributes": {"type": "object"},
            "finding_description": {
                "type": "string",
                "description": (
                    "A clear, specific description of the visual finding "
                    "(e.g. 'Right lower lobe consolidation with air bronchograms')."
                ),
            },
            "image_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "List of image IDs from the image store that support this finding. "
                    "Include the most relevant images — cropped ROIs, annotated views, "
                    "enhanced versions, and/or probe images."
                ),
                "minItems": 1,
            },
            "confidence": {
                "type": "number",
                "description": (
                    "Confidence score from 0.0 to 1.0 indicating how certain you are "
                    "about this finding based on the visual evidence."
                ),
                "minimum": 0.0,
                "maximum": 1.0,
            },
        },
        "required": ["finding_description", "image_evidence_ids", "confidence"],
    }
    is_concurrency_safe = False  # modifies shared pipeline state
    is_read_only = False

    def prompt(self) -> str:
        return (
            "Use link_visual_evidence as the FINAL step for each finding you identify. "
            "After you have cropped, enhanced, annotated, and probed the relevant region, "
            "call this tool to formally record the finding with its supporting image evidence. "
            "Every finding MUST be linked to at least one image_id. The confidence score "
            "should reflect both the clarity of the visual evidence and the likelihood of "
            "the finding. A confidence below 0.3 suggests the finding is uncertain and "
            "should be noted as such."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        finding_description: str = args.get("finding_description", "").strip()
        image_evidence_ids: list[str] = args.get("image_evidence_ids", [])
        confidence: float = float(args.get("confidence", 0.0))

        # --- Validation ---
        if not finding_description:
            return ToolResult(
                data=json.dumps({"error": "finding_description must not be empty"}),
                is_error=True,
            )

        if not image_evidence_ids:
            return ToolResult(
                data=json.dumps({"error": "image_evidence_ids must contain at least one image ID"}),
                is_error=True,
            )

        confidence = max(0.0, min(1.0, confidence))

        pipeline_state: PipelineState = context.extra.get("pipeline_state")
        image_store = context.extra.get("image_store")

        if pipeline_state is None or image_store is None:
            return ToolResult(
                data=json.dumps({"error": "pipeline_state or image_store not found in context"}),
                is_error=True,
            )

        # Verify all referenced images exist
        missing = []
        valid_ids = []
        for img_id in image_evidence_ids:
            if image_store.get(img_id) is None:
                missing.append(img_id)
            else:
                valid_ids.append(img_id)

        if missing:
            return ToolResult(
                data=json.dumps({
                    "error": f"Image(s) not found in store: {missing}. Cannot link non-existent evidence.",
                }),
                is_error=True,
            )

        previous = next((o for o in pipeline_state.observation_graph if o.get("observation_id") == args.get("observation_id")), None)
        if args.get("observation_id") and previous is None:
            return ToolResult(data=json.dumps({"error": "Unknown observation_id"}), is_error=True)
        lower = finding_description.lower()
        uncertain = bool(re.search(r"\b(possible|possibly|may|might|cannot exclude|suspected|questionable)\b", lower))
        negated = bool(re.search(r"\b(no|without|absent|negative for)\b", lower)) and not uncertain
        laterality = "bilateral" if "bilateral" in lower else "left" if re.search(r"\bleft\b", lower) else "right" if re.search(r"\bright\b", lower) else "unspecified"
        obs = {
            "original_text": finding_description,
            "anatomy": args.get("anatomy", (previous or {}).get("anatomy", "")),
            "laterality": args.get("laterality", (previous or {}).get("laterality", laterality)),
            "polarity": args.get("polarity", "uncertain" if uncertain else "negative" if negated else "positive"),
            "certainty": args.get("certainty", "uncertain" if uncertain else "stated"),
            "attributes": args.get("attributes", {}),
            "confidence": confidence, "image_evidence_ids": valid_ids,
        }
        obs["observation_id"] = args.get("observation_id") or stable_id("OBS", [finding_description, sorted(valid_ids)])
        try:
            prepare_record_update(pipeline_state, obs["observation_id"], "observation", obs)
        except ValueError as exc:
            return ToolResult(data=json.dumps({"error": str(exc)}), is_error=True)
        if previous:
            pipeline_state.observation_graph[pipeline_state.observation_graph.index(previous)] = obs
        elif not any(o.get("observation_id") == obs["observation_id"] for o in pipeline_state.observation_graph):
            pipeline_state.observation_graph.append(obs)
        graph = getattr(pipeline_state, "dependency_graph", None)
        if graph is not None:
            graph.add_node(obs["observation_id"], "observation", obs)
            for img_id in valid_ids:
                graph.add_node(img_id, "image", image_store.get_metadata(img_id))
                graph.add_edge(img_id, obs["observation_id"])
            graph.set_dependencies(obs["observation_id"], valid_ids)

        # Build the tool chain from all referenced images
        all_tools: list[str] = []
        for img_id in valid_ids:
            chain = image_store.get_tool_chain(img_id)
            for tool_name in chain:
                if tool_name not in all_tools:
                    all_tools.append(tool_name)
        # Always include this tool
        if self.name not in all_tools:
            all_tools.append(self.name)

        # Remove superseded bindings for this observation only; other observations
        # using the same source image retain their independent bindings.
        if previous:
            pipeline_state.image_evidence[:] = [e for e in pipeline_state.image_evidence
                if getattr(e, "observation_id", "") != obs["observation_id"] or e.image_id in valid_ids]
        # Create evidence records — one per image, all sharing the same
        # finding description and confidence
        created_evidence: list[dict] = []
        for img_id in valid_ids:
            _, img_desc = image_store.get(img_id)
            img_meta = image_store._metadata.get(img_id, {})
            evidence = ImageEvidence(
                image_id=img_id,
                description=img_desc,
                tool_chain=image_store.get_tool_chain(img_id) + [self.name],
                finding_description=finding_description,
                confidence=confidence,
                crop_box=img_meta.get("crop_box"),
                observation_id=obs["observation_id"],
            )
            if hasattr(evidence, "metadata"):
                evidence.metadata = {**img_meta, "observation_id": obs["observation_id"]}
            existing = next((e for e in pipeline_state.image_evidence if e.image_id == img_id and getattr(e, "observation_id", "") == obs["observation_id"]), None)
            if existing:
                pipeline_state.image_evidence[pipeline_state.image_evidence.index(existing)] = evidence
            else:
                pipeline_state.image_evidence.append(evidence)
            created_evidence.append(evidence.to_dict())

        # Build a human-readable summary
        evidence_count = len(pipeline_state.image_evidence)
        confidence_label = (
            "high" if confidence >= 0.7
            else "moderate" if confidence >= 0.4
            else "low"
        )

        result = {
            "status": "linked",
            "observation_id": obs["observation_id"],
            "finding_description": finding_description,
            "confidence": confidence,
            "confidence_label": confidence_label,
            "evidence_count_for_finding": len(valid_ids),
            "total_evidence_count": evidence_count,
            "tool_chain": all_tools,
            "linked_images": [
                {"image_id": e["image_id"], "description": e["description"]}
                for e in created_evidence
            ],
            "message": (
                f"Finding linked with {confidence_label} confidence ({confidence:.2f}): "
                f"'{finding_description}' backed by {len(valid_ids)} image(s). "
                f"Total evidence records in pipeline: {evidence_count}."
            ),
        }

        logger.info(
            "Linked evidence: '%s' (conf=%.2f) <- %s",
            finding_description[:60], confidence, valid_ids,
        )
        return ToolResult(
            data=json.dumps(result),
            metadata={
                "finding_description": finding_description,
                "observation_id": obs["observation_id"],
                "confidence": confidence,
                "consumed_ids": valid_ids,
                "produced_ids": [obs["observation_id"]],
            },
        )
