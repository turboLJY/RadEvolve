"""Create diagnostic claims linked to actual finding claims."""
import re
from src.core.tool import Tool, ToolUseContext
from src.core.types import Impression, ToolResult, ValidationResult
from src.report.grounding import CLAIM_PROPERTIES, ensure_mutable, resolve_support, sync_claim_graph


def _strip_leading_number(text: str) -> str:
    return re.sub(r"^\d+[\.\)]\s*", "", text)


class BuildImpressionTool(Tool):
    name = "build_structured_impression"
    description = "Create one impression claim or replace an affected impression using its existing impression_id"
    is_read_only = False
    input_schema = {"type": "object", "properties": {
        "text": {"type": "string"},
        "impression_id": {"type": "string"},
        "type": {"type": "string", "enum": ["diagnosis", "recommendation", "differential"]},
        "linked_finding_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "image_evidence_ids": {"type": "array", "items": {"type": "string"}},
        "retrieved_evidence_ids": {"type": "array", "items": {"type": "string"}},
        **CLAIM_PROPERTIES,
    }, "required": ["text", "linked_finding_ids"]}

    def prompt(self):
        return ("Create one independently testable impression per call, linked to existing findings. "
                "Evidence can be inherited through those findings. Recommendations require applicable knowledge. "
                "For repair, replace affected impression_id and preserve all unaffected claims.")

    def validate_input(self, args):
        if not isinstance(args.get("text"), str) or not args["text"].strip():
            return ValidationResult(False, "Impression text cannot be empty")
        if not args.get("linked_finding_ids"):
            return ValidationResult(False, "Must link to at least one finding")
        if not 0 <= args.get("confidence", 0.0) <= 1:
            return ValidationResult(False, "Confidence must be between zero and one")
        return ValidationResult(True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None or state.structured_report is None:
            return ToolResult("Error: create supporting findings first", is_error=True)
        validation = self.validate_input(args)
        if not validation.valid:
            return ToolResult(validation.message, is_error=True)
        try:
            report = state.structured_report
            linked = list(dict.fromkeys(args["linked_finding_ids"]))
            unknown = set(linked) - {f.finding_id for f in report.findings}
            if unknown:
                raise ValueError(f"Unknown supporting findings: {sorted(unknown)}")
            support = resolve_support(state, args)
            identifier = args.get("impression_id")
            old = next((i for i in report.impressions if i.impression_id == identifier), None)
            if identifier:
                if old is None:
                    raise ValueError(f"Unknown impression ID: {identifier}")
                ensure_mutable(state, identifier)
            else:
                identifier = f"I{max([int(i.impression_id[1:]) for i in report.impressions if i.impression_id[1:].isdigit()] + [0])+1:03d}"
                if state.iteration_count > 1:
                    if not set(linked) & state.affected_node_ids:
                        raise ValueError("New repair impressions must depend on affected findings")
                    state.affected_node_ids.add(identifier)
            fields = {name: args[name] for name in CLAIM_PROPERTIES if name in args and name != "prior_examination_ids"}
            fields.setdefault("observation_ids", list(dict.fromkeys(e.observation_id for e in support.image_evidence if e.observation_id)))
            fields.setdefault("claim_type", "recommendation" if args.get("type") == "recommendation" else "diagnostic")
            impression = Impression(impression_id=identifier, text=_strip_leading_number(args["text"].strip()),
                                    type=args.get("type", "diagnosis"), linked_findings=linked,
                                    confidence=args.get("confidence", 0.0), support=support,
                                    generating_operation_id=context.extra.get("operation_id", ""), **fields)
            if old:
                report.impressions[report.impressions.index(old)] = impression
            else:
                report.impressions.append(impression)
            sync_claim_graph(state, impression)
            return ToolResult(f"Impression {identifier} stored: {impression.to_dict()}", metadata={"produced_ids": [identifier]})
        except (ValueError, TypeError) as exc:
            return ToolResult(f"Error: {exc}", is_error=True)
