"""Synthetic fixtures for software verification only; not clinical evidence."""
from __future__ import annotations

from dataclasses import replace
from .engine import EvolutionEngine, TargetDefect
from .evaluation import Assessment, ReferenceFacts, RunOutput, PairedEvaluation
from .manifest import DevelopmentCase, PatientManifest
from .schemas import Checkpoint, HarnessEdit, write_json


def synthetic_pair(case_id='synthetic_dev_1', patient_id='synthetic_patient_1', repetition=0):
    reference = ReferenceFacts(case_id, patient_id, ({'fact_id': 'R1', 'text': 'Synthetic reference fact one'},
                                                  {'fact_id': 'R2', 'text': 'Synthetic reference fact two'}))
    common = {'reference_version': reference.version, 'covered_fact_ids': frozenset({'R1', 'R2'}),
              'unverifiable_claim_ids': frozenset(), 'target_required_fact_ids': frozenset({'R1'})}
    control = Assessment(**common,
        clinical_errors=({'error_id': 'synthetic_error', 'severity': 'major', 'kind': 'H5_attribute_error', 'confirmed': True},),
        evidence_relations=({'claim_id': 'F1', 'relation': 'wrong'}, {'claim_id': 'F2', 'relation': 'fully_supported'}),
        correct_claim_ids=frozenset({'F2'}), target_defect_present=True)
    candidate = Assessment(**common, clinical_errors=(),
        evidence_relations=({'claim_id': 'F1', 'relation': 'fully_supported'}, {'claim_id': 'F2', 'relation': 'fully_supported'}),
        correct_claim_ids=frozenset({'F1', 'F2'}), target_defect_present=False)
    run = RunOutput({'synthetic_only': True}, True, 1., True)
    return PairedEvaluation(reference, repetition, control, candidate, run, run, frozenset({'F2'}))


class SyntheticRunner:
    async def checkpoint(self, case_input, stage, harness, skills, versions):
        return Checkpoint(case_input['case_id'], stage, versions, [], {}, [], [], {'nodes': {}, 'edges': []}, [], {'synthetic_only': True})

    async def run(self, case_input, harness, skills, versions, checkpoint=None):
        return RunOutput({'synthetic_only': True, 'edited': bool(harness.accepted_edits)}, True, 1., True)


class SyntheticEvaluator:
    async def evaluate(self, reference, target, control, candidate):
        pair = synthetic_pair(reference.case_id, reference.patient_id)
        first = pair.candidate if control.report['edited'] else pair.control
        second = pair.candidate if candidate.report['edited'] else pair.control
        return first, second, first.correct_claim_ids & second.correct_claim_ids


class SyntheticProposer:
    async def propose(self, target, harness, max_candidates):
        return [HarnessEdit.from_dict({'edit_id': 'synthetic_edit', 'stage': 'report_writing',
            'field': 'task_instructions', 'value': 'Synthetic software demonstration rule.',
            'target_defect': target.defect_id, 'assessment_criteria': target.assessment_criteria,
            'preserve': ['Synthetic fact R2'], 'source_case_ids': [target.case_id]})]

    async def skills(self, target, accepted_edit, harness, max_candidates):
        return []


async def run_demo(output_dir):
    cases = []
    for index, role in enumerate(['evolution', 'development', 'development']):
        pair = synthetic_pair(f'synthetic_case_{index}', f'synthetic_patient_{index}')
        cases.append(DevelopmentCase(pair.reference.case_id, pair.reference.patient_id, 'SYNTHETIC_ONLY',
                                    'train', role, 'SYNTHETIC_NO_IMAGE', {'modality': 'CXR'}, pair.reference.facts))
    manifest = PatientManifest(tuple(cases))
    target = TargetDefect('synthetic_attribute_error', cases[0].case_id, {'synthetic_only': True}, {}, [],
                          'Synthetic fixture requires fact R1 to remain represented.',
                          {case.case_id: ['R1'] for case in cases})
    engine = EvolutionEngine(manifest=manifest, runner=SyntheticRunner(), evaluator=SyntheticEvaluator(),
        proposer=SyntheticProposer(), permitted_tools={'report_writing': []},
        fixed_versions={'model': 'SYNTHETIC', 'evaluator': 'SYNTHETIC', 'retrieval_snapshot': 'SYNTHETIC', 'resource_limits': 'SYNTHETIC'},
        reference_cost_cny=1., output_dir=output_dir)
    result = await engine.run([target])
    result['synthetic_only'] = True
    result['purpose'] = 'Software verification only; no patients, model API calls, or measured clinical outcomes.'
    write_json(engine.output_dir / 'demo_result.json', result)
    return result
