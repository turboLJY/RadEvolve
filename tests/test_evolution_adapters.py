"""No-API adapter tests covering checkpoint restoration and evaluator completeness."""
from dataclasses import asdict
from types import SimpleNamespace
import json

import pytest
from PIL import Image

from src.core.types import PipelineState, StructuredReport, Finding, EventType, StreamEvent
from src.data.image_store import SessionImageStore
from src.evolution.runner import PipelineRunner, encode_state, decode_state, encode_image_store, decode_image_store
from src.evolution.schemas import Harness, SkillLibrary, Versions, content_hash, write_json
from src.evolution.evaluation import RunOutput
from src.evolution.models import ModelPairedEvaluator, ModelProposer, JSONModel
from src.evolution.engine import TargetDefect
from src.evolution.demo import synthetic_pair
from src.evolution.manifest import DevelopmentCase, PatientManifest
from src.llm.client import ModelConfig
from scripts.run_evolution import baseline_reference_cost


def runner_settings(tmp_path):
    path=tmp_path/'corpus.json'
    write_json(path,{'schema_version':1,'sources':['radlex','snomed','guideline','case'],'records':[],'excluded_patient_ids':[]})
    return {'retrieval':{'snapshot_path':str(path)},'pipeline':{'max_iterations':3},'limits':{'max_image_views':32}}


async def fake_pipeline(image,context,settings,client,*,initial_state=None,image_store=None,start_stage=None,runtime_bundle=None,checkpoint_callback=None):
    state=initial_state or PipelineState(session_id='synthetic',original_image=image,exam_info=context,iteration_count=1)
    store=image_store or SessionImageStore(state.session_id)
    if not initial_state:
        store.save(image,'Synthetic image','input');store._coarse_inspected={('sequence_0','axial')}
    assert ('sequence_0','axial') in store._coarse_inspected
    stages=['visual_analysis','evidence_retrieval','report_writing','quality_review']
    start=stages.index(start_stage) if start_stage else 0
    for stage in stages[start:]:
        if checkpoint_callback: checkpoint_callback(stage,state,store)
        request={'request_id':stage,'cost_cny':.1,'complete':True}
        state.request_records.append(request)
        state.conversation_history[stage]=[{'role':'assistant','content':'Synthetic stage'}]
        yield StreamEvent(EventType.STATUS,{'request_record':request})
    state.structured_report=StructuredReport(findings=[Finding('F1','Synthetic finding')])
    result=state.to_dict();result.update(completed=True,iterations=1,images={})
    yield StreamEvent(EventType.DONE,result)


def make_runner(tmp_path,pipeline=fake_pipeline):
    settings=runner_settings(tmp_path)
    runner=PipelineRunner(settings,SimpleNamespace(config=ModelConfig()),pipeline=pipeline)
    harness=Harness();skills=SkillLibrary()
    versions=Versions(runner.model_version,harness.version,skills.version,runner.snapshot_version,'fixed_eval',runner.resource_version)
    image=tmp_path/'case.png';Image.new('RGB',(8,8)).save(image)
    inputs={'case_id':'synthetic','patient_id':'synthetic','image_path':str(image),'clinical_context':{'modality':'CXR'}}
    return runner,harness,skills,versions,inputs


@pytest.mark.asyncio
async def test_pipeline_runner_restores_json_checkpoint_with_images_and_upstream_history(tmp_path):
    runner,harness,skills,versions,inputs=make_runner(tmp_path)
    checkpoint=await runner.checkpoint(inputs,'report_writing',harness,skills,versions)
    state=decode_state(checkpoint.state['pipeline'])
    assert len(state.request_records)==2 and state.original_image.size==(8,8)
    restored_store=decode_image_store(checkpoint.state['image_store'])
    assert restored_store._coarse_inspected=={('sequence_0','axial')}
    run=await runner.run(inputs,harness,skills,versions,checkpoint.clone())
    assert run.complete and run.cost_cny==pytest.approx(.4)
    assert run.trace['executed_cost_cny']==pytest.approx(.2)
    assert len(decode_state(checkpoint.state['pipeline']).request_records)==2
    assert len(run.report['request_records'])==4


@pytest.mark.asyncio
async def test_source_change_and_fixed_model_change_reject_checkpoint(tmp_path):
    runner,harness,skills,versions,inputs=make_runner(tmp_path)
    checkpoint=await runner.checkpoint(inputs,'report_writing',harness,skills,versions)
    Image.new('RGB',(8,8),'red').save(inputs['image_path'])
    with pytest.raises(ValueError,match='different case inputs'):
        await runner.run(inputs,harness,skills,versions,checkpoint)
    runner.llm_client.config.temperature=.22
    with pytest.raises(ValueError,match='model changed'):
        await runner.run(inputs,harness,skills,versions)


@pytest.mark.asyncio
async def test_failed_pipeline_keeps_incurred_request_costs(tmp_path):
    async def broken(*args,**kwargs):
        yield StreamEvent(EventType.STATUS,{'request_record':{'request_id':'failed','cost_cny':.15,'complete':False}})
        raise RuntimeError('synthetic failure')
    runner,harness,skills,versions,inputs=make_runner(tmp_path,broken)
    output=await runner.run(inputs,harness,skills,versions)
    assert not output.complete and output.cost_cny==.15
    assert output.trace['errors'][0]['error_type']=='RuntimeError'


def test_checkpoint_codec_rejects_arbitrary_object_types():
    with pytest.raises(ValueError,match='Prohibited'):
        decode_state({'__type__':'os.system','value':'untrusted'})
    with pytest.raises(TypeError,match='Unsupported'):
        encode_state(SimpleNamespace(x=1))


def model_pair():
    pair=synthetic_pair();report={'structured_report':{'findings':[{'finding_id':'F1','text':'One'},{'finding_id':'F2','text':'Two'}],'impressions':[]},'versions':{'harness':'SECRET_ARM_LABEL'},'images':{}}
    run=RunOutput(report,True,1.,True)
    target=TargetDefect('defect',pair.reference.case_id,{}, {}, [],'fixed criteria',{pair.reference.case_id:['R1']})
    return pair,run,target


@pytest.mark.asyncio
async def test_model_evaluator_blinds_versions_and_requires_all_claim_relations():
    pair,run,target=model_pair()
    class FakeModel:
        version='synthetic'
        async def complete(self,prompt,payload,images):
            assert 'SECRET_ARM_LABEL' not in json.dumps(payload)
            self.payload=payload
            return {'A':pair.control.to_dict(),'B':pair.candidate.to_dict(),'retained_A_in_B':['F2'],'retained_B_in_A':['F2']}
    model=FakeModel();evaluator=ModelPairedEvaluator(model)
    first,second,retained=await evaluator.evaluate(pair.reference,target,run,run)
    assert retained==frozenset({'F2'})
    class IncompleteModel(FakeModel):
        async def complete(self,*args):
            result=await super().complete(*args);result['A']['evidence_relations']=[];return result
    with pytest.raises(ValueError,match='every generated structured claim'):
        await ModelPairedEvaluator(IncompleteModel()).evaluate(pair.reference,target,run,run)


@pytest.mark.asyncio
async def test_proposer_never_receives_development_reference_mapping():
    pair,run,target=model_pair()
    target.required_fact_ids['development_case']=['DEV_SECRET']
    class FakeModel:
        version='synthetic'
        async def complete(self,prompt,payload):
            assert 'DEV_SECRET' not in json.dumps(payload)
            return {'edits':[]}
    proposer=ModelProposer(FakeModel(),FakeModel(),{})
    assert await proposer.propose(target,Harness())==[]


def test_structured_model_retry_budget_enforced():
    with pytest.raises(ValueError,match='five retries'):
        JSONModel(None,model='m',temperature=0,top_p=1,max_retries=6)


def test_baseline_cost_requires_same_cases_repeats_patients_and_budget(tmp_path):
    manifest=PatientManifest((DevelopmentCase('e','ep','d','train','evolution','x',{},({'fact_id':'R','text':'synthetic'},)),
                              DevelopmentCase('v','vp','d','train','development','x',{},({'fact_id':'R','text':'synthetic'},))))
    versions={'model':'m','retrieval_snapshot':'r','harness':'h','skills':'s','resource_limits':'b'}
    data={'versions':versions,'patient_manifest_hash':content_hash(manifest.to_dict()),'runs':[
        {'case_id':'v','patient_id':'vp','repetition':i,'complete':True,'cost_cny':1.} for i in range(3)]}
    path=tmp_path/'costs.json';write_json(path,data)
    args=(path,'m','r','h','s','b',manifest,3)
    assert baseline_reference_cost(*args)==1.
    data['runs'].pop();write_json(path,data)
    with pytest.raises(ValueError,match='required complete'): baseline_reference_cost(*args)
    data['runs'][0]['patient_id']='wrong';write_json(path,data)
    with pytest.raises(ValueError,match='source-patient'): baseline_reference_cost(*args)
