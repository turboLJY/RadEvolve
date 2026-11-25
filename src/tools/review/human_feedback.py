"""Human feedback injection tool for doctor collaboration."""

from src.core.tool import Tool, ToolUseContext
from src.core.types import ToolResult


class HumanFeedbackTool(Tool):
    name = "inject_human_feedback"
    description = "Allow a doctor to provide feedback on the report (interactive mode only)"
    input_schema = {
        "type": "object",
        "properties": {
            "feedback_text": {
                "type": "string",
                "description": "Doctor's feedback text",
            },
            "override_verdict": {
                "type": "string",
                "enum": ["pass", "revise"],
                "description": "Override the automated review verdict",
            },
        },
        "required": [],
    }

    def prompt(self) -> str:
        return (
            "Use inject_human_feedback to incorporate doctor feedback. "
            "In batch mode this is a no-op. In interactive mode, it pauses "
            "for human input."
        )

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        feedback = args.get("feedback_text", "")
        override = args.get("override_verdict")

        if override:
            return ToolResult(data="Review verdicts are determined by recorded claim review and integrity checks; tool-supplied overrides are not accepted.", is_error=True)

        if feedback:
            return ToolResult(data=f"Doctor feedback recorded: {feedback}")

        return ToolResult(data="No feedback provided. Continuing with automated review.")