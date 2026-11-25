"""Deterministically render structured claim text with lossless span mappings."""
from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult
from src.report.grounding import render_claims


class RenderReportTool(Tool):
    name = "render_natural_language"
    description = "Render exact claim text into FINDINGS and IMPRESSION and retain claim span IDs"
    is_read_only = False
    input_schema = {"type": "object", "properties": {"style": {"type": "string", "enum": ["standard", "detailed", "brief"]}}, "required": []}

    def prompt(self):
        return "Render after assembling the report. Every output span comes from a structured claim; do not append new clinical prose."

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None or state.structured_report is None:
            return ToolResult("Error: no report to render", is_error=True)
        text = render_claims(state)
        return ToolResult(text, metadata={"claim_spans": state.rendered_claim_spans})
