"""Build or locally replace one independently testable finding claim."""
from src.core.tool import Tool, ToolUseContext
from src.core.types import Finding, StructuredReport, ToolResult, ValidationResult
from src.report.grounding import CLAIM_PROPERTIES, ensure_mutable, resolve_support, sync_claim_graph


class BuildFindingTool(Tool):
    name = "build_structured_finding"
    description = "Create one finding claim or replace an affected finding using its existing finding_id"
    is_read_only = False
    input_schema = {
        "type": "object", "properties": {
            "text": {"type": "string"},
            "finding_id": {"type": "string", "description": "Existing affected ID when repairing; omit for a new claim."},
            "type": {"type": "string", "enum": ["normal", "abnormal", "uncertain"]},
            "observation": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "image_evidence_ids": {"type": "array", "items": {"type": "string"}},
            "retrieved_evidence_ids": {"type": "array", "items": {"type": "string"}},
            **CLAIM_PROPERTIES,
        }, "required": ["text", "anatomy"],
    }

    def prompt(self):
        return ("Create one independently testable proposition per claim. Preserve anatomy, laterality, polarity, "
                "attributes and certainty. Every patient-specific observation, including normal and negative claims, "
                "requires current-examination visual evidence. Comparisons also require supplied prior examinations; "
                "recommendations require applicable knowledge. During repair pass existing finding_id to replace only "
                "affected claims; never recreate unaffected claims or use similar cases as proof of patient findings.")

    def validate_input(self, args):
        if not isinstance(args.get("text"), str) or not args["text"].strip():
            return ValidationResult(False, "Finding text cannot be empty")
        if not 0 <= args.get("confidence", 0.0) <= 1:
            return ValidationResult(False, "Confidence must be between zero and one")
        return ValidationResult(True)

    async def call(self, args: dict, context: ToolUseContext) -> ToolResult:
        state = context.extra.get("pipeline_state")
        if state is None:
            return ToolResult("Error: no pipeline state", is_error=True)
        validation = self.validate_input(args)
        if not validation.valid:
            return ToolResult(validation.message, is_error=True)
        try:
            support = resolve_support(state, args)
            report = state.structured_report or StructuredReport()
            identifier = args.get("finding_id")
            old = next((f for f in report.findings if f.finding_id == identifier), None)
            if identifier:
                if old is None:
                    raise ValueError(f"Unknown finding ID: {identifier}")
                ensure_mutable(state, identifier)
            else:
                identifier = f"F{max([int(f.finding_id[1:]) for f in report.findings if f.finding_id[1:].isdigit()] + [0])+1:03d}"
                if state.iteration_count > 1:
                    dependencies = set(args.get("observation_ids", []) + args.get("image_evidence_ids", []) + args.get("retrieved_evidence_ids", []))
                    if not (dependencies & state.affected_node_ids) and not any(i.omitted_observation_id for i in state.repair_issues):
                        raise ValueError("New repair claims must depend on an affected observation/evidence or address a recorded omission")
                    state.affected_node_ids.add(identifier)
            fields = {name: args[name] for name in CLAIM_PROPERTIES if name in args and name != "prior_examination_ids"}
            fields.setdefault("observation_ids", list(dict.fromkeys(e.observation_id for e in support.image_evidence if e.observation_id)))
            finding = Finding(finding_id=identifier, text=args["text"].strip(), type=args.get("type", "abnormal"),
                              observation=args.get("observation", ""), confidence=args.get("confidence", 0.0), support=support,
                              generating_operation_id=context.extra.get("operation_id", ""), **fields)
            if old:
                report.findings[report.findings.index(old)] = finding
            else:
                report.findings.append(finding)
            state.structured_report = report
            sync_claim_graph(state, finding)
            return ToolResult(f"Finding {identifier} stored: {finding.to_dict()}", metadata={"produced_ids": [identifier]})
        except (ValueError, TypeError) as exc:
            return ToolResult(f"Error: {exc}", is_error=True)
