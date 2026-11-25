"""OpenAI-compatible proposal, skill-generation and blinded evaluation adapters."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
import time

from .engine import TargetDefect
from .evaluation import Assessment, ReferenceFacts, RunOutput
from .prompts import PROMPT_VERSION, PROPOSER_PROMPT, SKILL_PROMPT, EVALUATOR_PROMPT
from .schemas import Harness, HarnessEdit, Skill, content_hash, strict_keys


class JSONModel:
    def __init__(self, client, *, model: str, temperature: float, top_p: float,
                 max_tokens: int = 8192, max_retries: int = 5):
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or not 0 <= max_retries <= 5:
            raise ValueError('Structured model calls permit zero to five retries')
        self.client = client
        self.model, self.temperature, self.top_p = model, temperature, top_p
        self.max_tokens, self.max_retries = max_tokens, max_retries
        self.records: list[dict] = []

    @property
    def version(self) -> str:
        return content_hash({'model': self.model, 'temperature': self.temperature, 'top_p': self.top_p,
                             'max_tokens': self.max_tokens, 'max_retries': self.max_retries})

    async def complete(self, system: str, payload: dict, images: list[dict] | None = None) -> dict:
        from src.llm.client import _is_retryable
        blocks = [{'type': 'text', 'text': json.dumps(payload, ensure_ascii=False)}]
        for image in images or []:
            blocks.extend([{'type': 'text', 'text': image['label']},
                           {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + image['base64']}}])
        for attempt in range(self.max_retries + 1):
            start = time.monotonic()
            record = {'model': self.model, 'attempt': attempt, 'prompt_version': PROMPT_VERSION,
                      'prompt_hash': content_hash(system), 'complete': False}
            try:
                response = await self.client.chat.completions.create(
                    model=self.model, temperature=self.temperature, top_p=self.top_p,
                    max_completion_tokens=self.max_tokens, response_format={'type': 'json_object'},
                    messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': blocks}])
                choice = response.choices[0]
                record['usage'] = response.usage.model_dump() if response.usage else None
                record['finish_reason'] = choice.finish_reason
                if choice.finish_reason != 'stop':
                    raise ValueError('Incomplete structured model response')
                result = json.loads(choice.message.content or '')
                if not isinstance(result, dict):
                    raise ValueError('Model response is not a JSON object')
                record['complete'] = True
                return result
            except Exception as exc:
                record['error_type'] = type(exc).__name__
                if attempt >= self.max_retries or not _is_retryable(exc):
                    raise
                await asyncio.sleep(min(30, .5 * (2 ** attempt)))
            finally:
                record['elapsed_seconds'] = time.monotonic() - start
                self.records.append(record)
        raise RuntimeError('Model attempts exhausted')


class ModelProposer:
    def __init__(self, proposal_model: JSONModel, skill_model: JSONModel, permitted_tools: dict[str, list[str]]):
        self.proposal_model, self.skill_model = proposal_model, skill_model
        self.permitted_tools = permitted_tools

    @property
    def version(self) -> str:
        return content_hash({'proposal': self.proposal_model.version, 'skills': self.skill_model.version,
                             'prompts': [PROMPT_VERSION, PROPOSER_PROMPT, SKILL_PROMPT]})

    @staticmethod
    def _source_target(target: TargetDefect) -> dict:
        value = asdict(target)
        value['required_fact_ids'] = {target.case_id: target.required_fact_ids[target.case_id]}
        return value

    async def propose(self, target: TargetDefect, harness: Harness, max_candidates: int = 4) -> list[HarnessEdit]:
        result = await self.proposal_model.complete(PROPOSER_PROMPT, {
            'target': self._source_target(target), 'harness': harness.to_dict(), 'permitted_tools': self.permitted_tools,
            'max_candidates': min(max_candidates, 4)})
        strict_keys(result, {'edits'}, {'edits'})
        if not isinstance(result['edits'], list) or len(result['edits']) > min(max_candidates, 4):
            raise ValueError('Invalid proposal count')
        return [HarnessEdit.from_dict(edit) for edit in result['edits']]

    async def skills(self, target: TargetDefect, accepted_edit: HarnessEdit, harness: Harness,
                     max_candidates: int = 2) -> list[Skill]:
        result = await self.skill_model.complete(SKILL_PROMPT, {
            'target': self._source_target(target), 'accepted_edit': accepted_edit.to_dict(), 'harness': harness.to_dict(),
            'permitted_tools': self.permitted_tools, 'max_candidates': min(max_candidates, 2)})
        strict_keys(result, {'skills'}, {'skills'})
        if not isinstance(result['skills'], list) or len(result['skills']) > min(max_candidates, 2):
            raise ValueError('Invalid skill proposal count')
        return [Skill.from_dict(skill) for skill in result['skills']]


def _blind_report(run: RunOutput) -> dict:
    # Do not expose harness/library IDs, prompts, trace metadata or arm names.
    report = run.report
    structured = report.get('structured_report') or {}
    return {'findings': structured.get('findings', []), 'impressions': structured.get('impressions', []),
            'natural_language_report': report.get('natural_language_report', ''),
            'image_evidence': report.get('image_evidence', []),
            'retrieved_evidence': report.get('retrieved_evidence', [])}


def _claim_ids(report: dict) -> set[str]:
    claims = report['findings'] + report['impressions']
    result = [claim.get('finding_id') or claim.get('impression_id') for claim in claims]
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError('Evaluator requires nonempty uniquely identified structured claims')
    return set(result)


class ModelPairedEvaluator:
    def __init__(self, model: JSONModel):
        self.model = model
        self._pair_counter = 0

    @property
    def version(self) -> str:
        return content_hash({'model': self.model.version, 'prompt': EVALUATOR_PROMPT, 'version': PROMPT_VERSION})

    async def evaluate(self, reference: ReferenceFacts, target: TargetDefect,
                       control: RunOutput, candidate: RunOutput):
        if not control.complete or not candidate.complete:
            raise ValueError('Incomplete reports cannot receive acceptance adjudication')
        reports = [_blind_report(control), _blind_report(candidate)]
        # Alternation hides version labels and avoids a fixed preferred arm.
        swap = bool(self._pair_counter % 2)
        self._pair_counter += 1
        order = [1, 0] if swap else [0, 1]
        runs = [control, candidate]
        payload = {'reference': asdict(reference), 'reference_version': reference.version,
                   'target_defect': target.defect_id, 'assessment_criteria': target.assessment_criteria,
                   'target_required_fact_ids': target.required_fact_ids[reference.case_id],
                   'reports': {label: reports[index] for label, index in zip(('A', 'B'), order)}}
        images = []
        for label, index in zip(('A', 'B'), order):
            for image_id, image in runs[index].report.get('images', {}).items():
                if image.get('base64'):
                    images.append({'label': f'{label}: {image_id}', 'base64': image['base64']})
        result = await self.model.complete(EVALUATOR_PROMPT, payload, images)
        strict_keys(result, {'A', 'B', 'retained_A_in_B', 'retained_B_in_A'}, {'A', 'B', 'retained_A_in_B', 'retained_B_in_A'})
        assessments = [Assessment.from_dict(result[label]) for label in ('A', 'B')]
        for index, assessment in zip(order, assessments):
            expected = _claim_ids(reports[index])
            assessed = {relation['claim_id'] for relation in assessment.evidence_relations}
            if expected != assessed:
                raise ValueError('Evidence adjudication must cover every generated structured claim exactly in scope')
            if not assessment.correct_claim_ids <= expected or not assessment.unverifiable_claim_ids <= expected:
                raise ValueError('Assessment contains unknown generated claim IDs')
            if assessment.target_required_fact_ids != frozenset(target.required_fact_ids[reference.case_id]):
                raise ValueError('Evaluator changed fixed target requirements')
            assessment.validate_reference(reference)
        control_index = 1 if swap else 0
        candidate_index = 0 if swap else 1
        retained = result['retained_B_in_A' if swap else 'retained_A_in_B']
        from .schemas import string_list
        retained = frozenset(string_list(retained, 'retained_control_claim_ids'))
        if not retained <= assessments[control_index].correct_claim_ids:
            raise ValueError('Retention includes a control claim not judged correct')
        return assessments[control_index], assessments[candidate_index], retained
