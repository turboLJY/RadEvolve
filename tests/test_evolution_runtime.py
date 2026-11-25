"""Final freeze rejects actual configuration drift before any reporting call."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from src.agents.base import build_agent
from src.core.types import AgentRole, PipelineState
from src.evolution.manifest import PatientManifest, freeze_manifest
from src.evolution.runtime import (capture_reporting_contract, check_runtime, load_runtime,
                                   verify_agent_config, RuntimeBundle)
from src.evolution.schemas import Harness, SkillLibrary, content_hash, write_json
from src.kb.snapshot import CorpusSnapshot
from src.llm.client import ModelConfig


@pytest.fixture
def frozen_runtime(tmp_path):
    snapshot = CorpusSnapshot.materialize(tmp_path / "corpus.json", [], ["radlex", "snomed", "case", "guideline"])
    settings = {
        "retrieval": {"snapshot_path": str(tmp_path / "corpus.json"), "snapshot_version": snapshot.version, "require_snapshot": True},
        "embedding": {"model_name": "local-model", "device": "cpu"},
        "pipeline": {"max_iterations": 3}, "agents": {}, "limits": {"max_image_views": 32}, "accounting": {},
    }
    model = ModelConfig(model_name="frozen-model", api_key="NOT_A_REAL_KEY")
    client = SimpleNamespace(config=model)
    agents = {role.value: build_agent(role, settings, PipelineState()) for role in AgentRole}
    contract = capture_reporting_contract(settings, agents)
    harness, skills = Harness(), SkillLibrary()
    model_dict = asdict(model)
    model_dict.pop("api_key")
    versions = {"model": content_hash(model_dict), "harness": harness.version, "skills": skills.version,
                "retrieval_snapshot": snapshot.version, "evaluator": "fixed-evaluator",
                "resource_limits": content_hash({key: settings.get(key, {}) for key in ("pipeline", "agents", "limits", "accounting")})}
    files = {
        "harness": harness.to_dict(), "skills": skills.to_dict(),
        "prompts": {"reporting_contract": contract}, "tool_schemas": contract["tool_schemas"],
        "evaluation_settings": {}, "patient_manifest": PatientManifest(()).to_dict(),
    }
    components = {"retrieval_snapshot": tmp_path / "corpus.json"}
    for name, value in files.items():
        components[name] = tmp_path / f"{name}.json"
        write_json(components[name], value)
    freeze_manifest(tmp_path / "frozen.json", components, versions, {}, PatientManifest(()))
    settings["evolution"] = {"frozen_manifest_path": str(tmp_path / "frozen.json"),
                             "harness_path": str(components["harness"]), "skill_library_path": str(components["skills"])}
    return settings, client, agents, versions


def test_frozen_actual_configuration_and_case_context_are_verified(frozen_runtime):
    settings, client, agents, versions = frozen_runtime
    bundle = load_runtime(settings)
    check_runtime(bundle, settings, client, agents)
    assert bundle.versions == versions
    # The case's data may change while the reporting template remains fixed.
    state = PipelineState(modality="brain_mri", exam_info={"sequence": "T2", "clinical_indication": "Headache"})
    config = build_agent(AgentRole.VISUAL_ANALYSIS, settings, state)
    verify_agent_config(bundle, settings, config, state)


@pytest.mark.parametrize("component", ["model", "resources", "snapshot", "hooks", "embedding", "live_retrieval"])
def test_freeze_rejects_execution_drift(frozen_runtime, component):
    settings, client, _, _ = frozen_runtime
    settings, client = deepcopy(settings), deepcopy(client)
    bundle = load_runtime(settings)
    if component == "model":
        client.config.model_name = "different-model"
    elif component == "resources":
        settings["limits"]["max_image_views"] = 16
    elif component == "snapshot":
        settings["retrieval"]["snapshot_version"] = "wrong-version"
    elif component == "hooks":
        settings["hooks"] = {"min_findings": 0}
    elif component == "embedding":
        settings["embedding"]["model_name"] = "different-embedding"
    else:
        settings["retrieval"]["require_snapshot"] = False
    with pytest.raises(ValueError):
        check_runtime(bundle, settings, client)


def test_actual_stage_cannot_change_prompt_permissions_or_parameters(frozen_runtime):
    settings, client, _, _ = frozen_runtime
    bundle = load_runtime(settings)
    check_runtime(bundle, settings, client)
    state = PipelineState()
    config = build_agent(AgentRole.REPORT_WRITING, settings, state)
    config.system_prompt += "\nUnfrozen instruction"
    with pytest.raises(ValueError, match="case prompt"):
        verify_agent_config(bundle, settings, config, state)
    config = build_agent(AgentRole.REPORT_WRITING, settings, state)
    config.temperature = 1.5
    with pytest.raises(ValueError, match="parameters"):
        verify_agent_config(bundle, settings, config, state)
    config = build_agent(AgentRole.REPORT_WRITING, settings, state)
    tool = config.tool_registry.all_tools()[0]
    tool.input_schema = {"type": "object", "properties": {"unfrozen": {"type": "string"}}}
    with pytest.raises(ValueError, match="permissions"):
        verify_agent_config(bundle, settings, config, state)


def test_api_key_rotation_does_not_change_model_identity(frozen_runtime):
    settings, client, _, _ = frozen_runtime
    client.config.api_key = "DIFFERENT_TEST_KEY"
    check_runtime(load_runtime(settings), settings, client)


def test_unfrozen_development_remains_supported():
    check_runtime(RuntimeBundle(), {}, SimpleNamespace())


@pytest.mark.asyncio
async def test_reporting_entry_rejects_frozen_drift_before_model_call(frozen_runtime):
    from PIL import Image
    from src.core.orchestrator import run_pipeline
    settings, client, _, _ = frozen_runtime
    client.config.model_name = "unvalidated-replacement"
    with pytest.raises(ValueError, match="reporting model"):
        _ = [event async for event in run_pipeline(Image.new("RGB", (8, 8)),
                                                  {"modality": "CXR"}, settings, client)]


def test_skill_required_inputs_accept_zero_and_false_and_explicit_observation_types():
    from src.evolution.runtime import select_skill
    from src.evolution.schemas import Skill
    skill = Skill("normal-check", "report_writing", {"claim_type": ["normality"]},
                  ("exam_info.age", "exam_info.contrast"),
                  ({"tool": "build_structured_finding", "instruction": "Check visual support"},),
                  ("Check current-study evidence",), (), 1, "accepted-edit", ("evolution-case",),
                  status="active", validation_record={"accepted": True}, tested_scope={"claim_type": ["normality"]})
    state = PipelineState(exam_info={"age": 0, "contrast": False},
                          observation_graph=[{"observation_id": "OBS1", "claim_type": "normality"}])
    selected, reason = select_skill(SkillLibrary((skill,)), "report_writing", state, ["build_structured_finding"])
    assert selected == skill and reason == "applicable"
    state.exam_info["contrast"] = None
    assert select_skill(SkillLibrary((skill,)), "report_writing", state, ["build_structured_finding"])[0] is None
    state.exam_info["contrast"] = False
    state.observation_graph = [{"observation_id": "OBS1", "original_text": "No effusion", "polarity": "negative"}]
    assert select_skill(SkillLibrary((skill,)), "report_writing", state, ["build_structured_finding"])[0] is None
