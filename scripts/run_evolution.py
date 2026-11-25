#!/usr/bin/env python3
"""Offline evolution, deterministic admission audit, and synthetic smoke test."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import math
from pathlib import Path
from statistics import mean
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evolution.evaluation import AcceptanceSettings, PairedEvaluation, acceptance, local_screen
from src.evolution.schemas import write_json, content_hash


def acceptance_settings(settings: dict) -> AcceptanceSettings:
    cfg = settings.get('evolution', {})
    if cfg.get('loss_weights', [.5, .25, .25]) != [.5, .25, .25]:
        raise ValueError('This implementation fixes the paper loss weights at [0.50, 0.25, 0.25]')
    return AcceptanceSettings(
        local_repetitions=cfg.get('local_repetitions', 5), validation_repetitions=cfg.get('validation_repetitions', 3),
        bootstrap_resamples=cfg.get('bootstrap_samples', 20000), bootstrap_seed=cfg.get('bootstrap_seed', 20260615),
        confidence=cfg.get('confidence_level', .95), min_gain=cfg.get('min_loss_gain', .01),
        min_retention=1-cfg.get('retention_tolerance', .01), min_coverage_difference=-cfg.get('coverage_tolerance', .01),
        cost_multiplier=cfg.get('max_cost_ratio', 1.25))


def baseline_reference_cost(path, expected_model, expected_corpus, initial_harness, initial_skills,
                            expected_resource_limits, manifest, repetitions):
    """Initial costs use the same validation cases, repetitions and patient weights."""
    data = json.loads(Path(path).read_text())
    expected = {'model': expected_model, 'retrieval_snapshot': expected_corpus,
                'harness': initial_harness, 'skills': initial_skills, 'resource_limits': expected_resource_limits}
    if data.get('versions') != expected:
        raise ValueError('Baseline costs do not describe this initial model/corpus/h0/S0/resource configuration')
    if data.get('patient_manifest_hash') != content_hash(manifest.to_dict()):
        raise ValueError('Baseline costs were measured with a different patient manifest')
    expected_cases = {case.case_id: case for case in manifest.subset('development')}
    expected_runs = {(case_id, k) for case_id in expected_cases for k in range(repetitions)}
    observed = set()
    grouped = {}
    for row in data.get('runs', []):
        cost = row.get('cost_cny')
        key = (row.get('case_id'), row.get('repetition'))
        if key not in expected_runs or key in observed:
            raise ValueError('Baseline contains duplicate or unexpected case/repetition records')
        observed.add(key)
        if row.get('patient_id') != expected_cases[key[0]].patient_id:
            raise ValueError('Baseline source-patient identity differs from the development cohort')
        if not row.get('complete') or not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            raise ValueError('Baseline requires complete, priced runs and stable patient IDs')
        grouped.setdefault(row['patient_id'], []).append(cost)
    if observed != expected_runs or not grouped:
        raise ValueError('Initial reference cost lacks required complete development repetitions')
    return mean(mean(costs) for costs in grouped.values())


async def real_run(args):
    import yaml
    from src.core.types import AgentRole, PipelineState
    from src.agents.base import build_agent
    from src.llm.client import LLMClient, ModelConfig, set_shared_client
    from src.evolution.engine import EvolutionEngine, TargetDefect
    from src.evolution.manifest import PatientManifest, freeze_manifest
    from src.evolution.models import JSONModel, ModelProposer, ModelPairedEvaluator
    from src.evolution.runner import PipelineRunner
    from src.evolution.runtime import load_runtime
    from src.evolution.schemas import Harness, SkillLibrary
    from src.evolution.prompts import PROMPT_VERSION, PROPOSER_PROMPT, SKILL_PROMPT, EVALUATOR_PROMPT
    from src.kb.snapshot import CorpusSnapshot

    settings = yaml.safe_load(Path(args.settings).read_text()) or {}
    manifest = PatientManifest.from_dict(json.loads(Path(args.manifest).read_text()))
    targets = [TargetDefect(**item) for item in json.loads(Path(args.defects).read_text())]
    if not targets:
        raise ValueError('At least one prespecified defect is required')
    cfg = settings.get('evolution', {})
    if cfg.get('max_candidates_per_defect', 4) != 4 or cfg.get('max_skills_per_intervention', 2) != 2:
        raise ValueError('Candidate limits are fixed at four harness edits and two skills per intervention')
    prices = settings.get('accounting', {})
    if any(prices.get(name) is None for name in ('input_cny_per_million', 'output_cny_per_million')):
        raise ValueError('Archived input/output CNY prices are required; no current price is assumed')
    settings.setdefault('retrieval', {})['require_snapshot'] = True
    client = LLMClient(ModelConfig.from_settings(args.settings))
    set_shared_client(client)
    runner = PipelineRunner(settings, client)
    snapshot = CorpusSnapshot(settings['retrieval']['snapshot_path'])
    final_patients = {case.patient_id for case in manifest.subset('final', evaluable_only=False)}
    if any(record.get('patient_id') in final_patients for record in snapshot.records if record['source_type'] == 'case'):
        raise ValueError('Final-evaluation patients are present in the retrieval corpus')
    bundle = load_runtime(settings)
    settings['retrieval']['snapshot_version'] = snapshot.version
    # Inspect permissions from actual registries rather than granting model-written names.
    permissions, schemas, initial_agent_configs = {}, {}, {}
    for role in AgentRole:
        agent = build_agent(role, settings, PipelineState())
        initial_agent_configs[role.value] = agent
        permissions[role.value] = [tool.name for tool in agent.tool_registry.all_tools()]
        schemas[role.value] = agent.tool_registry.get_api_schemas()
    def model(role, default_name, temperature, top_p):
        options = cfg.get(role, {})
        return JSONModel(client.client, model=options.get('model', default_name),
                         temperature=options.get('temperature', temperature), top_p=options.get('top_p', top_p),
                         max_tokens=options.get('max_tokens', 8192), max_retries=settings.get('model', {}).get('max_retries', 5))
    proposal = model('proposer', 'qwen3.6-flash', .7, .95)
    skill = model('skill_generator', 'qwen3.6-flash', .2, .95)
    evaluator_model = model('evaluator', 'qwen3.6-max-preview', 0., 1.)
    evaluator = ModelPairedEvaluator(evaluator_model)
    proposer = ModelProposer(proposal, skill, permissions)
    protocol = acceptance_settings(settings)
    reference_cost = baseline_reference_cost(args.baseline_costs, runner.model_version, runner.snapshot_version,
                                            Harness().version, SkillLibrary().version, runner.resource_version, manifest, protocol.validation_repetitions)
    engine = EvolutionEngine(manifest=manifest, runner=runner, evaluator=evaluator, proposer=proposer,
        permitted_tools=permissions, fixed_versions={'model': runner.model_version, 'retrieval_snapshot': runner.snapshot_version,
            'evaluator': evaluator.version, 'resource_limits': runner.resource_version},
        reference_cost_cny=reference_cost, output_dir=args.output_dir, harness=bundle.harness, skills=bundle.skills,
        settings=protocol, max_rounds=cfg.get('max_rounds', 5), max_candidate_evaluations=cfg.get('max_candidate_evaluations', 20))
    out = Path(args.output_dir)
    write_json(out/'patient_manifest.json', manifest.to_dict())
    write_json(out/'defects.json', [asdict(target) for target in targets])
    from src.evolution.runtime import capture_reporting_contract
    write_json(out/'prompts.json', {'version': PROMPT_VERSION, 'proposer': PROPOSER_PROMPT,
                                  'skill_generator': SKILL_PROMPT, 'evaluator': EVALUATOR_PROMPT,
                                  'reporting': {name: config.system_prompt for name, config in initial_agent_configs.items()},
                                  'reporting_contract': capture_reporting_contract(settings, initial_agent_configs)})
    model_config = asdict(client.config)
    model_config.pop('api_key', None)
    write_json(out/'model_configs.json', {'reporting': model_config,
               **{name: {'model': obj.model, 'temperature': obj.temperature, 'top_p': obj.top_p,
                         'max_tokens': obj.max_tokens, 'max_retries': obj.max_retries}
                  for name,obj in [('proposer',proposal),('skill_generator',skill),('evaluator',evaluator_model)]}})
    write_json(out/'tool_schemas.json', schemas)
    write_json(out/'evaluation_settings.json', asdict(protocol))
    try:
        result = await engine.run(targets)
    finally:
        write_json(out/'execution_accounting.json', {'reporting': runner.run_records, 'proposal': proposal.records,
                                                   'skill_generation': skill.records, 'evaluation': evaluator_model.records})
    frozen = freeze_manifest(out/'frozen_manifest.json', {
        'patient_manifest': out/'patient_manifest.json', 'prompts': out/'prompts.json', 'tool_schemas': out/'tool_schemas.json',
        'retrieval_snapshot': settings['retrieval']['snapshot_path'], 'harness': out/'harness.json', 'skills': out/'skills.json',
        'evaluation_settings': out/'evaluation_settings.json', 'defects': out/'defects.json',
        'model_configs': out/'model_configs.json'},
        engine.versions().to_dict(), asdict(protocol), manifest)
    print(json.dumps({'candidate_evaluations': result['candidate_evaluations'], 'harness': result['harness']['version'],
                      'skills': result['skills']['version'], 'frozen_manifest': str(out/'frozen_manifest.json')}, indent=2))


async def measure_baseline(args):
    """Measure h0/S0 under the same development cases and validation repetitions.

    This is a live model run. It never extracts reference text into generation
    context. It is separate from selection so B_ref cannot drift between rounds.
    """
    import yaml
    from src.llm.client import LLMClient, ModelConfig, set_shared_client
    from src.evolution.runner import PipelineRunner
    from src.evolution.manifest import PatientManifest
    from src.evolution.schemas import Harness, SkillLibrary, Versions
    settings = yaml.safe_load(Path(args.settings).read_text()) or {}
    manifest = PatientManifest.from_dict(json.loads(Path(args.manifest).read_text()))
    protocol = acceptance_settings(settings)
    client = LLMClient(ModelConfig.from_settings(args.settings))
    set_shared_client(client)
    runner = PipelineRunner(settings, client)
    harness, skills = Harness(), SkillLibrary()
    versions = Versions(runner.model_version, harness.version, skills.version, runner.snapshot_version,
                        'baseline-no-adjudication', runner.resource_version)
    result = {'schema_version': 1, 'patient_manifest_hash': content_hash(manifest.to_dict()),
              'versions': {key: value for key,value in versions.to_dict().items() if key != 'evaluator'},
              'repetitions': protocol.validation_repetitions, 'runs': []}
    for case in manifest.subset('development'):
        for repetition in range(protocol.validation_repetitions):
            run = await runner.run(case.generation_input(), harness, skills, versions)
            result['runs'].append({'case_id': case.case_id, 'patient_id': case.patient_id,
                                  'repetition': repetition, 'complete': run.complete,
                                  'cost_cny': run.cost_cny})
            write_json(args.output, result)
    write_json(str(args.output)+'.execution.json', runner.run_records)
    cost = baseline_reference_cost(args.output, runner.model_version, runner.snapshot_version,
                                   harness.version, skills.version, runner.resource_version,
                                   manifest, protocol.validation_repetitions)
    print(json.dumps({'reference_cost_cny': cost, 'measured_runs': len(result['runs']), 'output': args.output}))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    demo = commands.add_parser('demo', help='Synthetic software smoke test; makes no API calls')
    demo.add_argument('--output-dir', default='output/evolution_synthetic')
    audit = commands.add_parser('audit', help='Recompute deterministic gates from saved paired assessment records')
    audit.add_argument('--records', required=True, help='JSON list of PairedEvaluation records, or engine record with pairs')
    audit.add_argument('--reference-cost-cny', type=float, required=True)
    audit.add_argument('--case-ids', required=True, help='Comma-separated complete expected cohort')
    audit.add_argument('--local', action='store_true')
    audit.add_argument('--output', required=True)
    run = commands.add_parser('run', help='Run offline evolution with configured real model APIs')
    run.add_argument('--settings', default='config/settings.yaml')
    run.add_argument('--manifest', required=True)
    run.add_argument('--defects', required=True)
    run.add_argument('--baseline-costs', required=True)
    run.add_argument('--output-dir', required=True)
    baseline = commands.add_parser('measure-baseline', help='Live h0/S0 cost measurement on the complete development cohort')
    baseline.add_argument('--settings', default='config/settings.yaml')
    baseline.add_argument('--manifest', required=True)
    baseline.add_argument('--output', required=True)
    verify = commands.add_parser('verify-freeze', help='Verify all component hashes before final evaluation')
    verify.add_argument('manifest')
    return result


def main():
    args = parser().parse_args()
    if args.command == 'demo':
        from src.evolution.demo import run_demo
        result = asyncio.run(run_demo(args.output_dir))
        print(json.dumps({'synthetic_only': True, 'candidate_evaluations': result['candidate_evaluations'],
                          'accepted_edits': result['harness']['accepted_edits'], 'output_dir': args.output_dir}, indent=2))
    elif args.command == 'audit':
        raw = json.loads(Path(args.records).read_text())
        pairs = [PairedEvaluation.from_dict(pair) for pair in (raw.get('pairs', []) if isinstance(raw, dict) else raw)]
        case_ids = set(args.case_ids.split(','))
        if args.local:
            if len(case_ids) != 1:
                raise ValueError('Local screening requires exactly one evolution case')
            decision = local_screen(pairs, next(iter(case_ids)))
        else:
            decision = acceptance(pairs, case_ids, args.reference_cost_cny)
        write_json(args.output, decision.to_dict())
        print(json.dumps(decision.to_dict(), indent=2))
    elif args.command == 'measure-baseline':
        asyncio.run(measure_baseline(args))
    elif args.command == 'verify-freeze':
        from src.evolution.manifest import verify_freeze
        print(json.dumps({'verified': True, 'manifest_hash': verify_freeze(args.manifest)['manifest_hash']}))
    else:
        asyncio.run(real_run(args))


if __name__ == '__main__':
    main()
