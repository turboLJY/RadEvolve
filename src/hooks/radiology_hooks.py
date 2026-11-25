"""
RadClaw-specific quality hooks for the agentic loop.
These enforce medical report quality before allowing an agent to finish.
"""

from __future__ import annotations

from src.core.types import LoopState
from src.hooks.engine import Hook, HookEvaluation


class FindingCompletenessHook(Hook):
    name = "finding_completeness"
    description = "Ensure at least N findings are documented"

    def __init__(self, min_findings: int = 2):
        self.min_findings = min_findings

    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        pipeline = kwargs.get("pipeline_state")
        if pipeline is not None and not pipeline.image_evidence:
            return HookEvaluation(passed=False, feedback="Document the actual observations, including normal findings, with link_visual_evidence and existing image IDs before finishing.")
        return HookEvaluation(passed=True)



class EvidenceSupportHook(Hook):
    name = "evidence_support"
    description = "Ensure findings reference supporting evidence"

    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        pipeline_state = kwargs.get("pipeline_state")
        if pipeline_state and pipeline_state.image_evidence:
            unsupported = [
                e for e in pipeline_state.image_evidence
                if not e.finding_description
            ]
            if len(unsupported) > len(pipeline_state.image_evidence) // 2:
                return HookEvaluation(
                    passed=False,
                    feedback="Many image evidence items lack finding descriptions. Use link_visual_evidence to bind them.",
                )
        return HookEvaluation(passed=True)


class ReportStructureHook(Hook):
    name = "report_structure"
    description = "Ensure the report has both findings and impressions"

    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        pipeline_state = kwargs.get("pipeline_state")
        if pipeline_state and pipeline_state.structured_report is None:
            return HookEvaluation(passed=False, feedback="No structured report exists. Build supported findings and impressions before finishing.")
        if pipeline_state and pipeline_state.structured_report:
            report = pipeline_state.structured_report
            if not report.findings:
                return HookEvaluation(
                    passed=False,
                    feedback="The report has no findings. Use build_structured_finding to add findings.",
                )
            if not report.impressions:
                return HookEvaluation(
                    passed=False,
                    feedback="The report has no impressions. Use build_structured_impression to add impressions.",
                )
        return HookEvaluation(passed=True)


def build_visual_hooks(config: dict) -> list[Hook]:
    return [
        FindingCompletenessHook(min_findings=config.get("min_findings", 2)),
    ]


def build_evidence_hooks(config: dict) -> list[Hook]:
    return [EvidenceSupportHook()]


def build_report_hooks(config: dict) -> list[Hook]:
    return [ReportStructureHook()]


class ReviewCoverageHook(Hook):
    name = "review_coverage"
    async def evaluate(self, state: LoopState, **kwargs) -> HookEvaluation:
        pipeline = kwargs.get("pipeline_state")
        if pipeline is not None and not pipeline.review_completed:
            return HookEvaluation(False, "Use record_claim_review for every finding and impression, including evidence relations, and record all observations checked for omissions before ending review.")
        return HookEvaluation(True)


def build_review_hooks(config: dict) -> list[Hook]:
    return [ReviewCoverageHook()]