"""Attribute responsibility for an issue to the appropriate agent."""

from src.core.tool import Tool, ToolUseContext
from src.core.types import AgentRole, ReviewIssue, ToolResult, ValidationResult


ATTRIBUTION_MAP = {
    "visual": AgentRole.VISUAL_ANALYSIS,
    "retrieval": AgentRole.EVIDENCE_RETRIEVAL,
    "generation": AgentRole.REPORT_WRITING,
    "wording": AgentRole.REPORT_WRITING,
}


class AttributeIssueTool(Tool):
    name = "attribute_responsibility"
    description = "Attribute a discovered issue to the responsible agent in the pipeline"
    input_schema = {
        "type": "object",
        "properties": {
            "issue_description": {"type": "string", "description": "Description of the issue found"},
            "issue_type": {
                "type": "string",
                "enum": [
                    "visual_inconsistency", "missing_evidence", "overclaim",
                    "underclaim", "terminology_error", "internal_contradiction",
                    "confidence_miscalibration", "missing_anatomy", "omitted_finding",
                    "traceability_failure", "internal_inconsistency", "evidence_mismatch", "comparison_error",
                ],
                "description": "Type of issue",
            },
            "attribution": {
                "type": "string",
                "enum": ["visual", "retrieval", "generation", "wording"],
                "description": "Which pipeline stage is responsible",
            },
            "severity": {
                "type": "string",
                "enum": ["low", "medium", "high", "critical"],
                "description": "Issue severity",
            },
            "suggested_fix": {"type": "string", "description": "Suggested fix for the issue"},
            "related_finding_id": {"type": "string", "description": "Related finding or impression ID"},
            "omitted_observation_id": {"type": "string"},
            "evidence_mismatch": {"type": "string"},
            "implicated_operation": {"type": "string"},
        },
        "required": ["issue_description", "issue_type", "attribution", "severity"],
    }

    def prompt(self) -> str:
        return ("Attribute every issue to visual, retrieval, generation or wording. Include the affected "
                "claim ID or omitted observation ID, evidence mismatch and implicated operation when known. "
                "Missing links to available evidence are generation issues; missed image observations are visual. "
                "Normal and negative observations also require visual evidence. Remediation preserves unrelated claims.")

    def validate_input(self, args: dict) -> ValidationResult:
        if not args.get("issue_description", "").strip():
            return ValidationResult(valid=False, message="Issue description required")
        if args.get("attribution") not in ATTRIBUTION_MAP:
            return ValidationResult(valid=False, message=f"Attribution must be one of: {list(ATTRIBUTION_MAP.keys())}")
        return ValidationResult(valid=True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None:
            return ToolResult(data="Error: no pipeline state", is_error=True)

        from src.report.grounding import get_claim
        if args.get("related_finding_id") and get_claim(state, args["related_finding_id"]) is None:
            return ToolResult(data="Error: unknown affected claim", is_error=True)
        if args.get("omitted_observation_id") and not any(o.get("observation_id") == args["omitted_observation_id"] for o in state.observation_graph):
            return ToolResult(data="Error: unknown omitted observation", is_error=True)
        if args.get("implicated_operation") and args["implicated_operation"] not in state.dependency_graph.nodes:
            return ToolResult(data="Error: unknown implicated operation", is_error=True)
        issue_num = len(state.review_issues) + 1
        attribution = args["attribution"]

        issue = ReviewIssue(
            issue_id=f"RI{issue_num:03d}",
            issue_type=args["issue_type"],
            description=args["issue_description"],
            severity=args.get("severity", "medium"),
            attribution=attribution,
            target_agent=ATTRIBUTION_MAP[attribution],
            suggested_fix=args.get("suggested_fix", ""),
            related_claim_id=args.get("related_finding_id", ""),
            omitted_observation_id=args.get("omitted_observation_id", ""),
            evidence_mismatch=args.get("evidence_mismatch", ""),
            implicated_operation=args.get("implicated_operation", ""),
        )

        state.review_issues.append(issue)

        return ToolResult(
            data=(
                f"Issue {issue.issue_id} attributed:\n"
                f"  Type: {issue.issue_type}\n"
                f"  Description: {issue.description}\n"
                f"  Attribution: {issue.attribution} → {issue.target_agent.value}\n"
                f"  Severity: {issue.severity}\n"
                f"  Suggested fix: {issue.suggested_fix}"
            )
        )