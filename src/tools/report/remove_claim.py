"""Remove an affected unsupported claim without rewriting unrelated content."""
from src.core.tool import Tool
from src.core.types import ToolResult
from src.report.grounding import ensure_mutable, get_claim


class RemoveClaimTool(Tool):
    name = "remove_affected_claim"
    description = "Remove an affected unsupported claim; dependent impressions must then be corrected"
    is_read_only = False
    input_schema = {"type": "object", "properties": {"claim_id": {"type": "string"}, "reason": {"type": "string"}}, "required": ["claim_id", "reason"]}

    def prompt(self):
        return "Use only during remediation to remove an affected unsupported claim. Recompute its dependent impressions."

    async def call(self, args, context):
        state = context.extra.get("pipeline_state")
        if state is None or state.iteration_count <= 1:
            return ToolResult("Error: removal is only available during remediation", is_error=True)
        identifier = args.get("claim_id", "")
        try:
            ensure_mutable(state, identifier)
            target = get_claim(state, identifier)
            if target is None:
                raise ValueError("Unknown claim ID")
            state.affected_node_ids.update(state.dependency_graph.descendants({identifier}))
            report = state.structured_report
            if target in report.findings:
                report.findings.remove(target)
            else:
                report.impressions.remove(target)
            state.dependency_graph.add_node(identifier, "removed_claim", {"reason": args.get("reason", "")})
            state.repaired_node_ids.add(identifier)
            state.reviewed_claim_ids.discard(identifier)
            state.claim_reviews.pop(identifier, None)
            state.natural_language_report = ""
            state.rendered_claim_spans = []
            return ToolResult(f"Removed {identifier}; dependent claims require review", metadata={"produced_ids": [identifier]})
        except ValueError as exc:
            return ToolResult(f"Error: {exc}", is_error=True)
