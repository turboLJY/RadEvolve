"""Executable offline harness/skill evolution with injected reporting services.

The coordinator owns admission and state transitions. Proposers and evaluators
can suggest or assess content, but cannot bypass deterministic gates or change
frozen model/corpus/evaluation/resource settings.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol, Any

from .evaluation import (AcceptanceSettings, Assessment, Decision, PairedEvaluation, RunOutput,
                         acceptance, local_screen)
from .manifest import DevelopmentCase, PatientManifest
from .schemas import (Checkpoint, Harness, HarnessEdit, Skill, SkillLibrary, Versions,
                      content_hash, write_json)


@dataclass(frozen=True)
class TargetDefect:
    defect_id: str
    case_id: str
    review_finding: dict
    dependencies: dict
    remediation_trace: list
    assessment_criteria: str
    required_fact_ids: dict[str, list[str]]


class ReportingRunner(Protocol):
    async def checkpoint(self, case_input: dict, stage: str, harness: Harness,
                         skills: SkillLibrary, versions: Versions) -> Checkpoint: ...
    async def run(self, case_input: dict, harness: Harness, skills: SkillLibrary,
                  versions: Versions, checkpoint: Checkpoint | None = None) -> RunOutput: ...


class PairedEvaluator(Protocol):
    async def evaluate(self, reference, target: TargetDefect, control: RunOutput,
                       candidate: RunOutput) -> tuple[Assessment, Assessment, frozenset[str]]: ...


class Proposer(Protocol):
    async def propose(self, target: TargetDefect, harness: Harness, max_candidates: int) -> list[HarnessEdit]: ...
    async def skills(self, target: TargetDefect, accepted_edit: HarnessEdit, harness: Harness,
                     max_candidates: int) -> list[Skill]: ...


class EvolutionEngine:
    def __init__(self, *, manifest: PatientManifest, runner: ReportingRunner,
                 evaluator: PairedEvaluator, proposer: Proposer,
                 permitted_tools: dict[str, list[str]], fixed_versions: dict[str, str],
                 reference_cost_cny: float, output_dir: str | Path,
                 harness: Harness | None = None, skills: SkillLibrary | None = None,
                 settings: AcceptanceSettings = AcceptanceSettings(),
                 max_rounds: int = 5, max_candidate_evaluations: int = 20):
        # B_ref is deliberately not updated after an accepted candidate.
        if reference_cost_cny <= 0:
            raise ValueError('A measured positive initial complete-run cost is required')
        required = {'model', 'retrieval_snapshot', 'evaluator', 'resource_limits'}
        if set(fixed_versions) != required or any(not value for value in fixed_versions.values()):
            raise ValueError('All fixed execution version fingerprints are required')
        if max_rounds > 5 or max_rounds <= 0 or max_candidate_evaluations > 20 or max_candidate_evaluations <= 0:
            raise ValueError('Harness search exceeds the prespecified paper budget')
        if not manifest.subset('evolution') or not manifest.subset('development'):
            raise ValueError('Evolution and independent development-validation cohorts must be evaluable')
        self.manifest, self.runner, self.evaluator, self.proposer = manifest, runner, evaluator, proposer
        self.permitted_tools = deepcopy(permitted_tools)
        self.fixed_versions = deepcopy(fixed_versions)
        self.reference_cost_cny = reference_cost_cny
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.harness, self.skills = harness or Harness(), skills or SkillLibrary()
        self.settings, self.max_rounds, self.max_candidates = settings, max_rounds, max_candidate_evaluations
        self.candidate_evaluations = 0
        self.records: list[dict] = []
        self._checkpoint_cache: dict[tuple[str, str], Checkpoint] = {}
        self._persist_current()
        write_json(self.output_dir / 'selection_protocol.json', {
            'fixed_versions': self.fixed_versions, 'reference_cost_cny': self.reference_cost_cny,
            'settings': asdict(self.settings), 'max_rounds': max_rounds,
            'max_candidate_evaluations': max_candidate_evaluations,
            'patient_manifest': manifest.to_dict(), 'unevaluable_case_ids': manifest.unevaluable_case_ids,
            'permitted_tools': self.permitted_tools})

    def versions(self, harness: Harness | None = None, skills: SkillLibrary | None = None) -> Versions:
        return Versions(**self.fixed_versions, harness=(harness or self.harness).version,
                        skills=(skills or self.skills).version)

    def _persist_current(self) -> None:
        write_json(self.output_dir / 'versions' / f'{self.harness.version}.json', self.harness.to_dict())
        write_json(self.output_dir / 'versions' / f'{self.skills.version}.json', self.skills.to_dict())
        write_json(self.output_dir / 'harness.json', self.harness.to_dict())
        write_json(self.output_dir / 'skills.json', self.skills.to_dict())

    def _record(self, kind: str, decision: Decision, **details) -> dict:
        record = {'sequence': len(self.records), 'kind': kind, 'decision': decision.to_dict(),
                  'versions': self.versions().to_dict(), 'candidate_evaluations': self.candidate_evaluations, **details}
        record['record_id'] = content_hash(record)
        self.records.append(record)
        write_json(self.output_dir / 'records' / f'{record["sequence"]:04d}_{kind}.json', record)
        return record

    async def _checkpoint(self, case: DevelopmentCase, stage: str) -> Checkpoint:
        key = (case.case_id, stage)
        checkpoint = self._checkpoint_cache.get(key)
        if checkpoint is None or not checkpoint.compatible(self.versions()):
            checkpoint = await self.runner.checkpoint(case.generation_input(), stage, self.harness, self.skills, self.versions())
            if checkpoint.case_id != case.case_id or checkpoint.before_stage != stage or not checkpoint.compatible(self.versions()):
                raise ValueError('Reporting runner returned an incompatible pre-decision checkpoint')
            self._checkpoint_cache[key] = checkpoint.clone()
            write_json(self.output_dir / 'checkpoints' / f'{content_hash(key)[:16]}.json', checkpoint.to_dict())
        return checkpoint.clone()

    async def _pair(self, case: DevelopmentCase, target: TargetDefect, repetition: int,
                    candidate_harness: Harness, candidate_skills: SkillLibrary,
                    checkpoint: Checkpoint | None = None,
                    control_skills: SkillLibrary | None = None) -> PairedEvaluation:
        baseline_skills = control_skills or self.skills
        # Each call receives independent state; no mutated checkpoint crosses arms.
        control = await self.runner.run(case.generation_input(), self.harness, baseline_skills,
                                        self.versions(self.harness, baseline_skills), checkpoint.clone() if checkpoint else None)
        candidate = await self.runner.run(case.generation_input(), candidate_harness, candidate_skills,
                                          self.versions(candidate_harness, candidate_skills), checkpoint.clone() if checkpoint else None)
        first, second, retained = await self.evaluator.evaluate(case.reference, target, control, candidate)
        required = frozenset(target.required_fact_ids[case.case_id])
        if first.target_required_fact_ids != required or second.target_required_fact_ids != required:
            raise ValueError('Evaluator changed the prespecified correction requirements')
        return PairedEvaluation(case.reference, repetition, first, second, control, candidate, retained)

    async def _validation(self, target: TargetDefect, candidate_harness: Harness, candidate_skills: SkillLibrary,
                          *, control_skills: SkillLibrary | None = None) -> list[PairedEvaluation]:
        pairs = []
        for case in self.manifest.subset('development'):
            for repetition in range(self.settings.validation_repetitions):
                pairs.append(await self._pair(case, target, repetition, candidate_harness, candidate_skills,
                                              control_skills=control_skills))
        return pairs

    def _accept(self, pairs: list[PairedEvaluation], *, target_reduction: bool = True) -> Decision:
        return acceptance(pairs, {c.case_id for c in self.manifest.subset('development')},
                          self.reference_cost_cny, self.settings, require_target_reduction=target_reduction)

    def _validate_target(self, target: TargetDefect) -> None:
        cases = [*self.manifest.subset('evolution'), *self.manifest.subset('development')]
        required = {c.case_id for c in cases if c.allocation == 'development' or c.case_id == target.case_id}
        if not required <= target.required_fact_ids.keys():
            raise ValueError('Target-required reference facts must be fixed for every replay/validation case before selection')
        for case in cases:
            if case.case_id in required and not set(target.required_fact_ids[case.case_id]) <= case.reference.ids:
                raise ValueError('Target includes unknown reference fact IDs')

    async def evaluate_edit(self, edit: HarnessEdit, target: TargetDefect) -> Decision:
        if self.candidate_evaluations >= self.max_candidates:
            return Decision(False, ('search_budget_exhausted',))
        # Count entry into screening, including failures and re-evaluation.
        self.candidate_evaluations += 1
        records = []
        try:
            self._validate_target(target)
            self.manifest.require_sources(edit.source_case_ids)
            if edit.target_defect != target.defect_id or edit.assessment_criteria != target.assessment_criteria:
                raise ValueError('Candidate changed the prespecified target defect or assessment criteria')
            case = next(c for c in self.manifest.subset('evolution') if c.case_id == target.case_id)
            candidate = self.harness.apply(edit, self.permitted_tools)
            checkpoint = await self._checkpoint(case, edit.stage)
            local = [await self._pair(case, target, i, candidate, self.skills, checkpoint)
                     for i in range(self.settings.local_repetitions)]
            records = [p.to_dict() for p in local]
            screen = local_screen(local, case.case_id, self.settings)
            self._record('local_screen', screen, edit=edit.to_dict(), pairs=records)
            if not screen.accepted:
                return screen
            validation = await self._validation(target, candidate, self.skills)
            decision = self._accept(validation)
            record = self._record('harness_acceptance', decision, edit=edit.to_dict(),
                                  candidate_harness=candidate.to_dict(), pairs=[p.to_dict() for p in validation])
            if decision.accepted:
                self.harness = candidate
                self._checkpoint_cache.clear()
                self._persist_current()
                await self.regress_skills(target)
            return decision
        except Exception as exc:
            decision = Decision(False, ('candidate_execution_failed',), {'error_type': type(exc).__name__, 'message': str(exc)})
            self._record('harness_failure', decision, edit=edit.to_dict(), completed_pairs=records)
            return decision

    async def evaluate_skill(self, skill: Skill, target: TargetDefect) -> Decision:
        try:
            skill = Skill.from_dict(skill.to_dict())
            self.manifest.require_sources(skill.source_case_ids)
            if skill.source_edit_id not in self.harness.accepted_edits:
                raise ValueError('Skills must derive from an accepted harness intervention')
            if skill.status != 'candidate' or any(s.skill_id == skill.skill_id for s in self.skills.skills):
                raise ValueError('Skill admission requires a new candidate')
            if any(op['tool'] not in self.permitted_tools.get(skill.stage, []) for op in skill.operations):
                raise ValueError('Skill requests a tool outside fixed permissions')
            # Active only inside the isolated candidate arm until admission.
            trial = replace(skill, status='active', validation_record={'trial': True}, tested_scope=skill.applicability)
            candidate_library = self.skills.with_skill(trial)
            pairs = await self._validation(target, self.harness, candidate_library)
            decision = self._accept(pairs, target_reduction=False)
            record = self._record('skill_admission', decision, skill=skill.to_dict(), pairs=[p.to_dict() for p in pairs])
            status = 'active' if decision.accepted else 'rejected'
            stored = replace(skill, status=status, validation_record={'record_id': record['record_id'], 'decision': decision.to_dict()},
                             tested_scope={'applicability': skill.applicability, 'validation_cases': [c.case_id for c in self.manifest.subset('development')]})
            if decision.accepted:
                self.skills = self.skills.with_skill(stored)
                self._checkpoint_cache.clear()
                self._persist_current()
                await self.regress_skills(target)
            else:
                # Rejected proposals remain auditable without changing S_t.
                write_json(self.output_dir / 'rejected_skills' / f'{stored.version}.json', stored.to_dict())
            return decision
        except Exception as exc:
            decision = Decision(False, ('skill_execution_failed',), {'error_type': type(exc).__name__, 'message': str(exc)})
            self._record('skill_failure', decision, skill=skill.to_dict())
            rejected = replace(skill, status='rejected', deactivation_reason='skill_execution_failed')
            write_json(self.output_dir / 'rejected_skills' / f'{rejected.version}.json', rejected.to_dict())
            return decision

    async def regress_skills(self, target: TargetDefect) -> None:
        """Compare each retained skill with its removal under the current harness."""
        for skill in tuple(self.skills.skills):
            if skill.status != 'active':
                continue
            try:
                removed = self.skills.without(skill.skill_id)
                pairs = await self._validation(target, self.harness, self.skills, control_skills=removed)
                # Reapply original admission gates conservatively. Missing/incomplete
                # regression evidence cannot establish continued admissibility.
                decision = self._accept(pairs, target_reduction=False)
                record = self._record('skill_regression', decision, skill_id=skill.skill_id, pairs=[p.to_dict() for p in pairs])
            except Exception as exc:
                decision = Decision(False, ('regression_execution_failed',), {'error_type': type(exc).__name__, 'message': str(exc)})
                record = self._record('skill_regression', decision, skill_id=skill.skill_id)
            if not decision.accepted:
                disabled = replace(skill, status='disabled', deactivation_reason=';'.join(decision.reasons),
                                   validation_record={**skill.validation_record, 'regression_record_id': record['record_id']})
                self.skills = self.skills.with_skill(disabled)
                self._checkpoint_cache.clear()
                self._persist_current()

    async def run(self, defects: list[TargetDefect]) -> dict:
        for target in defects[:self.max_rounds]:
            if self.candidate_evaluations >= self.max_candidates:
                break
            self.manifest.require_sources((target.case_id,))
            self._validate_target(target)
            try:
                edits = await self.proposer.propose(target, self.harness, max_candidates=4)
                if len(edits) > 4:
                    raise ValueError('More than four candidates generated for one defect')
            except Exception as exc:
                self._record('proposal_failure', Decision(False, ('proposal_failed',), {'error_type': type(exc).__name__, 'message': str(exc)}))
                continue
            accepted_any = False
            for edit in edits:  # Fixed proposer order; every remaining edit sees latest h,S.
                if self.candidate_evaluations >= self.max_candidates:
                    break
                decision = await self.evaluate_edit(edit, target)
                if not decision.accepted:
                    continue
                accepted_any = True
                try:
                    supporting = next((record for record in reversed(self.records)
                                       if record['kind'] == 'local_screen' and record.get('edit', {}).get('edit_id') == edit.edit_id
                                       and record['decision']['accepted']), None)
                    if supporting is None:
                        raise ValueError('Accepted intervention lacks supporting evolution trajectories')
                    trajectories = []
                    evolution_ids = {case.case_id for case in self.manifest.subset('evolution')}
                    for pair in supporting['pairs']:
                        if pair['reference']['case_id'] not in evolution_ids:
                            raise ValueError('Skill derivation cannot use development-validation trajectories')
                        trajectories.append({'source': 'accepted_local_replay', 'case_id': pair['reference']['case_id'],
                                             'repetition': pair['repetition'], 'report': pair['candidate_run']['report'],
                                             'trace': pair['candidate_run']['trace']})
                    skill_target = replace(target, remediation_trace=[*deepcopy(target.remediation_trace), *trajectories])
                    skills = await self.proposer.skills(skill_target, edit, self.harness, max_candidates=2)
                    if len(skills) > 2:
                        raise ValueError('More than two skills generated per accepted intervention')
                    for skill in skills:
                        await self.evaluate_skill(skill, target)
                except Exception as exc:
                    self._record('skill_proposal_failure', Decision(False, ('skill_proposal_failed',), {'error_type': type(exc).__name__, 'message': str(exc)}), edit_id=edit.edit_id)
            if not accepted_any:
                break
        result = {'harness': self.harness.to_dict(), 'skills': self.skills.to_dict(),
                  'candidate_evaluations': self.candidate_evaluations, 'records': len(self.records),
                  'unevaluable_case_ids': self.manifest.unevaluable_case_ids}
        write_json(self.output_dir / 'selection_result.json', result)
        return result

    def rollback(self, harness: Harness, skills: SkillLibrary, reason: str) -> None:
        if not reason.strip():
            raise ValueError('Rollback requires an audit reason')
        for version in (harness.version, skills.version):
            if not (self.output_dir / 'versions' / f'{version}.json').is_file():
                raise ValueError('Rollback target was not recorded in this evolution history')
        self._record('rollback', Decision(True, ()), target_harness=harness.version, target_skills=skills.version, reason=reason)
        self.harness, self.skills = deepcopy(harness), deepcopy(skills)
        self._checkpoint_cache.clear()
        self._persist_current()
