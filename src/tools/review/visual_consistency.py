"""Deterministic visual binding checks; clinical agreement is recorded separately."""
from src.core.tool import Tool
from src.core.types import ToolResult
from src.report.grounding import integrity_errors, register_issue


class VisualConsistencyTool(Tool):
    name = "check_visual_consistency"
    description = "Check visual binding and applicability prerequisites for findings and impressions"
    input_schema = {"type": "object", "properties": {"finding_ids": {"type": "array", "items": {"type": "string"}}}, "required": []}

    def prompt(self):
        return "Check traceability prerequisites, then use record_claim_review to record clinical evidence agreement."

    async def call(self, args, context):
        state = context.extra.get("pipeline_state")
        if state is None or state.structured_report is None:
            return ToolResult("Error: no report to review", is_error=True)
        selected = set(args.get("finding_ids", []))
        errors = [e for e in integrity_errors(state) if (not selected or e[0] in selected)]
        errors = [e for e in errors if "visual evidence" in e[2]]
        for target, category, description in errors:
            register_issue(state, target, category, description)
        if errors:
            return ToolResult("Binding defects: " + "; ".join(f"{target}: {description}" for target, _, description in errors))
        return ToolResult("Binding prerequisites satisfied. Clinical support is not inferred from links; record each claim review.")
