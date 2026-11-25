"""Add validated support to an affected finding or impression."""
from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult
from src.report.grounding import ensure_mutable, get_claim, resolve_support, sync_claim_graph


class LinkSupportTool(Tool):
    name = "link_support_evidence"
    description = "Link existing image, retrieved or prior-examination evidence to a claim"
    is_read_only = False
    input_schema = {"type": "object", "properties": {
        "target_id": {"type": "string"},
        **{name: {"type": "array", "items": {"type": "string"}}
           for name in ("image_evidence_ids", "retrieved_evidence_ids", "prior_examination_ids")},
    }, "required": ["target_id"]}

    def prompt(self):
        return "Add only existing, applicable evidence IDs. A link records provenance; it does not establish clinical truth."

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None:
            return ToolResult("Error: no pipeline state", is_error=True)
        try:
            target = get_claim(state, args["target_id"])
            if target is None:
                raise ValueError("Unknown target claim")
            ensure_mutable(state, args["target_id"])
            support = resolve_support(state, args)
            for name in ("image_evidence", "retrieved_evidence", "prior_examination_ids"):
                existing = getattr(target.support, name)
                existing.extend(value for value in getattr(support, name) if value not in existing)
            sync_claim_graph(state, target)
            return ToolResult(f"Evidence linked to {args['target_id']}", metadata={"produced_ids": [args["target_id"]]})
        except (ValueError, KeyError) as exc:
            return ToolResult(f"Error: {exc}", is_error=True)
