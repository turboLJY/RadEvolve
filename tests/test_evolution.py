"""Synthetic regression tests for admission gates and version isolation."""
from dataclasses import replace
import json
import pytest
from src.evolution.demo import synthetic_pair, run_demo
from src.evolution.evaluation import Assessment, acceptance, local_screen
from src.evolution.manifest import DevelopmentCase, PatientManifest, allocate_training_patients, freeze_manifest, verify_freeze
from src.evolution.schemas import Harness, HarnessEdit, Versions, Checkpoint, write_json


def pairs(count=2):
    return [synthetic_pair(f'c{i}',f'p{i}',k) for i in range(count) for k in range(3)]


def test_loss_fixed_denominator_deduplication_and_no_clipping():
    p=synthetic_pair()
    a=replace(p.control,clinical_errors=p.control.clinical_errors*2+({'error_id':'omission','severity':'critical','kind':'omission','confirmed':True},),evidence_relations=p.control.evidence_relations*2)
    assert a.loss(p.reference)=={'clinical':.25,'evidence':.5,'omission':0.,'total':.25}
    errors=tuple({'error_id':f'e{i}','severity':'critical','kind':'H1','confirmed':True} for i in range(10))
    assert replace(a,clinical_errors=errors).loss(p.reference)['clinical']==5.
    a=replace(p.candidate,evidence_relations=({'claim_id':'F1','semantic_id':'same','relation':'partially_supported'},{'claim_id':'F2','semantic_id':'same','relation':'missing'}))
    assert a.loss(p.reference)['evidence']==.5


def test_unverifiable_claims_are_not_correct_or_confirmed_errors():
    p=synthetic_pair();data=p.control.to_dict();data['unverifiable_claim_ids']=['F2']
    with pytest.raises(ValueError,match='unverifiable'): Assessment.from_dict(data)
    errors=({'error_id':'u','severity':'major','kind':'H1','confirmed':False},)
    assert replace(p.control,clinical_errors=errors).loss(p.reference)['clinical']==0


def test_local_requires_five_completed_pairs_and_real_target_resolution():
    local=[synthetic_pair(repetition=i) for i in range(5)]
    assert local_screen(local,'synthetic_dev_1').accepted
    assert not local_screen(local[:-1],'synthetic_dev_1').accepted
    unchanged=[replace(p,candidate=p.control) for p in local]
    assert 'target_defect_not_reduced' in local_screen(unchanged,'synthetic_dev_1').reasons
    deleted=[replace(p,candidate=replace(p.candidate,covered_fact_ids=frozenset({'R2'}))) for p in local]
    assert 'target_defect_not_reduced' in local_screen(deleted,'synthetic_dev_1').reasons


def test_admission_paper_bootstrap_thresholds_and_fixed_cost_ceiling():
    decision=acceptance(pairs(),{'c0','c1'},1.)
    assert decision.accepted
    assert decision.metrics['lower_bounds']=={'gain':.25,'retention':1.,'coverage_difference':0.}
    assert decision.metrics['bootstrap']['bootstrap_resamples']==20000
    costly=[replace(p,candidate_run=replace(p.candidate_run,cost_cny=1.251)) for p in pairs()]
    assert 'cost_ceiling_exceeded' in acceptance(costly,{'c0','c1'},1.).reasons


def test_new_severe_error_and_severity_escalation_reject():
    records=pairs();error={'error_id':'new','severity':'major','kind':'H1','confirmed':True}
    records[0]=replace(records[0],candidate=replace(records[0].candidate,clinical_errors=(error,)))
    assert 'new_critical_or_high_severity_error' in acceptance(records,{'c0','c1'},1.).reasons
    p=synthetic_pair();escalated=replace(p,candidate=replace(p.candidate,clinical_errors=({**p.control.clinical_errors[0],'severity':'critical'},)))
    assert escalated.new_severe_errors=={'synthetic_error'}


def test_incomplete_unresolved_missing_cost_fail_closed():
    for field in ('complete','within_resource_limits'):
        records=pairs();records[0]=replace(records[0],candidate_run=replace(records[0].candidate_run,**{field:False}))
        assert not acceptance(records,{'c0','c1'},1.).accepted
    records=pairs();records[0]=replace(records[0],candidate=replace(records[0].candidate,unresolved_severe=True))
    assert not acceptance(records,{'c0','c1'},1.).accepted
    records=pairs();records[0]=replace(records[0],candidate_run=replace(records[0].candidate_run,cost_cny=None))
    assert 'missing_complete_run_cost' in acceptance(records,{'c0','c1'},1.).reasons
    assert not acceptance(pairs()[:-1],{'c0','c1'},1.).accepted


def test_retention_unestimable_or_lost_correct_claim_rejects():
    records=[replace(p,control=replace(p.control,correct_claim_ids=frozenset()),retained_control_claim_ids=frozenset()) for p in pairs()]
    assert 'unestimable_acceptance_metric' in acceptance(records,{'c0','c1'},1.).reasons
    records=[replace(p,retained_control_claim_ids=frozenset()) for p in pairs()]
    assert 'retention_lower_bound_failed' in acceptance(records,{'c0','c1'},1.).reasons


def test_patient_cluster_weight_is_not_run_weight():
    records=[]
    for p in pairs():
        ref=replace(p.reference,patient_id='shared')
        records.append(replace(p,reference=ref,control=replace(p.control,reference_version=ref.version),candidate=replace(p.candidate,reference_version=ref.version)))
    extra=[replace(synthetic_pair('c2','other',i),candidate_run=replace(synthetic_pair().candidate_run,cost_cny=1.2)) for i in range(3)]
    d=acceptance(records+extra,{'c0','c1','c2'},1.)
    assert d.accepted and d.metrics['patients']==2
    assert d.metrics['means']['candidate_cost']==pytest.approx(1.1)


def case(case_id='c0',patient='p0',allocation='evolution',partition='train',context=None):
    return DevelopmentCase(case_id,patient,'dataset',partition,allocation,'unused.png',context or {'modality':'CXR'},({'fact_id':'R1','text':'Synthetic fact'},))


def test_manifest_patient_leakage_and_input_labels_rejected():
    with pytest.raises(ValueError,match='leakage'): PatientManifest((case(),case('c1','p0','development')))
    with pytest.raises(ValueError,match='validation or test'): case(partition='test')
    with pytest.raises(ValueError,match='reporting-time'): case(context={'modality':'CXR','positive_labels':['disease']})
    with pytest.raises(ValueError,match='prohibited'): case(context={'modality':'CXR','acquisition':{'nested_labels':['disease']}})
    with pytest.raises(ValueError,match='leakage'): PatientManifest((case(allocation='retrieval'),case('final','p0','final','test')))


def test_ten_percent_training_allocation_disjoint_and_deterministic():
    rows=[{'dataset':'d','partition':'train','patient_id':f'p{i}'} for i in range(100)]
    result=allocate_training_patients(rows)
    assert len(result['evolution'])==len(result['development'])==10
    assert not set(result['evolution'])&set(result['development'])
    assert result==allocate_training_patients(rows)


def edit(**kwargs):
    return HarnessEdit.from_dict({'edit_id':'e1','stage':'report_writing','field':'task_instructions','value':'Preserve evidence.','target_defect':'d','assessment_criteria':'fixed','preserve':['R1'],'source_case_ids':['c0'],**kwargs})


def test_harness_single_field_permissions_and_content_hash():
    h0=Harness();h1=h0.apply(edit(),{'report_writing':['render_natural_language']})
    assert h1.parent_version==h0.version and h1.version!=h0.version
    assert h0.stages['report_writing']=={}
    assert Harness.from_dict(h1.to_dict()).version==h1.version
    with pytest.raises(ValueError): edit(field='model')
    with pytest.raises(ValueError,match='permissions'): h0.apply(edit(field='tool_order',value=['shell']),{'report_writing':[]})
    corrupt=h1.to_dict();corrupt['stages']['report_writing']['task_instructions']='Changed'
    with pytest.raises(ValueError,match='hash'): Harness.from_dict(corrupt)


def test_checkpoint_clones_are_isolated_and_versions_checked():
    versions=Versions('m','h','s','r','e','b');checkpoint=Checkpoint('c','report_writing',versions,[],{'a':[]},[],[],{},[],{})
    one,two=checkpoint.clone(),checkpoint.clone();one.working_memory['a'].append('change')
    assert not two.working_memory['a'] and not checkpoint.working_memory['a']
    assert not checkpoint.compatible(replace(versions,skills='new'))


@pytest.mark.asyncio
async def test_executable_synthetic_evolution(tmp_path):
    result=await run_demo(tmp_path)
    assert result['synthetic_only'] and result['harness']['accepted_edits']==['synthetic_edit']
    record=json.loads((tmp_path/'records/0001_harness_acceptance.json').read_text())
    assert len(record['pairs'])==6 and record['decision']['accepted']
    assert record['decision']['metrics']['bootstrap']['bootstrap_resamples']==20000


def test_freeze_detects_component_changes(tmp_path):
    manifest=PatientManifest((case(),case('c1','p1','development')));components={}
    for name in ('patient_manifest','prompts','tool_schemas','retrieval_snapshot','harness','skills','evaluation_settings'):
        path=tmp_path/f'{name}.json';write_json(path,{'synthetic':True});components[name]=path
    versions={'model':'m','harness':'h','skills':'s','retrieval_snapshot':'r','evaluator':'e','resource_limits':'b'}
    path=tmp_path/'frozen.json';freeze_manifest(path,components,versions,{},manifest)
    assert verify_freeze(path)['versions']==versions
    write_json(components['harness'],{'changed':True})
    with pytest.raises(ValueError,match='changed'): verify_freeze(path)


def test_missing_retention_patients_do_not_invalidate_estimable_cohort():
    records=pairs(count=4)
    records=[replace(p,control=replace(p.control,correct_claim_ids=frozenset()),retained_control_claim_ids=frozenset())
             if p.reference.patient_id!='p0' else p for p in records]
    decision=acceptance(records,{'c0','c1','c2','c3'},1.)
    assert decision.accepted
    assert decision.metrics['lower_bounds']['retention']==1.
    assert decision.metrics['estimable_bootstrap_resamples']['retention']<20000
    assert decision.metrics['estimable_bootstrap_resamples']['gain']==20000


@pytest.mark.asyncio
async def test_skill_derivation_uses_accepted_evolution_trajectories_only(tmp_path,monkeypatch):
    from src.evolution.demo import SyntheticProposer
    received=[]
    async def skills(self,target,accepted_edit,harness,max_candidates):
        received.append(target)
        return []
    monkeypatch.setattr(SyntheticProposer,'skills',skills)
    await run_demo(tmp_path)
    assert len(received)==1
    local=[row for row in received[0].remediation_trace if row.get('source')=='accepted_local_replay']
    assert len(local)==5
    assert {row['case_id'] for row in local}=={'synthetic_case_0'}


@pytest.mark.asyncio
async def test_skill_admission_regression_deactivation_and_rejection_preserves_library(tmp_path):
    from src.evolution.engine import EvolutionEngine,TargetDefect
    from src.evolution.demo import SyntheticRunner,SyntheticEvaluator,SyntheticProposer
    from src.evolution.schemas import Skill,SkillLibrary
    cases=[]
    for i,allocation in enumerate(['evolution','development','development']):
        p=synthetic_pair(f'c{i}',f'p{i}')
        cases.append(DevelopmentCase(f'c{i}',f'p{i}','synthetic','train',allocation,'unused.png',{},p.reference.facts))
    manifest=PatientManifest(tuple(cases))
    harness=Harness().apply(edit(),{'report_writing':['render_natural_language']})
    engine=EvolutionEngine(manifest=manifest,runner=SyntheticRunner(),evaluator=SyntheticEvaluator(),proposer=SyntheticProposer(),
        permitted_tools={'report_writing':['render_natural_language']},fixed_versions={'model':'m','retrieval_snapshot':'r','evaluator':'e','resource_limits':'b'},
        reference_cost_cny=1.,output_dir=tmp_path,harness=harness)
    target=TargetDefect('d','c0',{}, {}, [],'fixed',{c.case_id:['R1'] for c in cases})
    skill=Skill.from_dict({'skill_id':'candidate','stage':'report_writing','applicability':{'modality':['CXR']},
        'required_inputs':['image_evidence'],'operations':[{'tool':'render_natural_language','instruction':'Synthetic verification'}],
        'evidence_checks':['Synthetic support'],'exclusions':[],'priority':1,'source_edit_id':'e1','source_case_ids':['c0']})
    evidence=[synthetic_pair(c.case_id,c.patient_id,k) for c in cases[1:] for k in range(3)]
    async def beneficial(*args,**kwargs): return evidence
    engine._validation=beneficial
    assert (await engine.evaluate_skill(skill,target)).accepted
    assert engine.skills.skills[0].status=='active'
    active_version=engine.skills.version
    no_gain=[replace(p,candidate=p.control) for p in evidence]
    async def ineffective(*args,**kwargs): return no_gain
    engine._validation=ineffective
    rejected=replace(skill,skill_id='rejected')
    assert not (await engine.evaluate_skill(rejected,target)).accepted
    assert engine.skills.version==active_version
    assert len(list((tmp_path/'rejected_skills').glob('*.json')))==1
    severe={'error_id':'new_major','severity':'major','kind':'H1','confirmed':True}
    harmful=[replace(p,candidate=replace(p.candidate,clinical_errors=(severe,))) for p in evidence]
    async def harmful_runs(*args,**kwargs): return harmful
    engine._validation=harmful_runs
    await engine.regress_skills(target)
    assert engine.skills.skills[0].status=='disabled'
    assert 'new_critical_or_high_severity_error' in engine.skills.skills[0].deactivation_reason
