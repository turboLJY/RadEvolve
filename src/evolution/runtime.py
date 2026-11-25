"""Frozen per-case harness and deterministic, conditional skill invocation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field, is_dataclass
import json
import hashlib
from pathlib import Path
from typing import Any

from .schemas import Harness, Skill, SkillLibrary, content_hash


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _input(state: Any, path: str) -> Any:
    value = state
    for key in path.split('.'):
        value = _get(value, key)
        if value is None:
            return None
    return value


def scope_context(state: Any) -> dict[str, set[str]]:
    report = _get(state, 'structured_report')
    claims = list(_get(report, 'findings', []) or []) + list(_get(report, 'impressions', []) or [])
    observations = _get(state, 'observation_graph', []) or []
    evidence = set()
    if _get(state, 'image_evidence', []):
        evidence.add('visual')
    for item in _get(state, 'retrieved_evidence', []) or []:
        evidence.add('case' if _get(item, 'source_type') == 'case' else 'knowledge')
    exam = _get(state, 'exam_info', {})
    if _get(exam, 'prior_examinations') or _get(exam, 'prior_images'):
        evidence.add('prior')
    modalities = {_get(state, 'modality', '')}
    if 'CXR' in modalities:
        modalities.add('chest_xray')
    return {
        'modality': modalities - {''},
        'anatomy': {_get(item, 'anatomy', '') for item in [*claims, *observations]} - {''},
        # Only explicit, recognized claim types are eligible before writing;
        # polarity or a disease word does not imply a claim type.
        'claim_type': {_get(item, 'claim_type', _get(item, 'type', '')) for item in [*claims, *observations]}
                      & {'observation', 'normality', 'attribute', 'comparison', 'diagnostic', 'recommendation'},
        'defect_type': {_get(item, 'issue_type', '') for item in [*(_get(state, 'review_issues', []) or []), *(_get(state, 'repair_issues', []) or [])]} - {''},
        'evidence_availability': evidence,
    }


def _matches(condition: dict[str, list[str]], context: dict[str, set[str]]) -> bool:
    # AND across scope dimensions; OR among alternatives in one dimension.
    # Evidence availability lists are requirements, therefore all must exist.
    return all((set(values) <= context.get(key, set()) if key == 'evidence_availability'
                else bool(set(values) & context.get(key, set()))) for key, values in condition.items())


def _missing_input(value: Any) -> bool:
    return value is None or (isinstance(value, (str, list, tuple, set, frozenset, dict)) and len(value) == 0)


def select_skill(library: SkillLibrary, stage: str, state: Any, available_tools: list[str]) -> tuple[Skill | None, str]:
    context = scope_context(state)
    candidates = []
    for skill in library.skills:
        if skill.status != 'active' or skill.stage != stage:
            continue
        if not _matches(skill.applicability, context):
            continue
        if any(_matches(condition, context) for condition in skill.exclusions):
            continue
        if any(_missing_input(_input(state, name)) for name in skill.required_inputs):
            continue
        if any(operation['tool'] not in available_tools for operation in skill.operations):
            continue
        candidates.append(skill)
    if not candidates:
        return None, 'no_applicable_skill'
    priority = max(skill.priority for skill in candidates)
    preferred = [skill for skill in candidates if skill.priority == priority]
    if len(preferred) != 1:
        return None, 'unresolved_skill_conflict'
    return preferred[0], 'applicable'


@dataclass
class RuntimeBundle:
    harness: Harness = field(default_factory=Harness)
    skills: SkillLibrary = field(default_factory=SkillLibrary)
    frozen_versions: dict = field(default_factory=dict)
    invocations: list[dict] = field(default_factory=list)
    _selected: dict[tuple[int, str], Skill | None] = field(default_factory=dict, repr=False)
    frozen_manifest: dict | None = field(default=None, repr=False)

    @property
    def versions(self) -> dict:
        return {**self.frozen_versions, 'harness': self.harness.version, 'skills': self.skills.version}

    def stage_instructions(self, stage: str, state: Any, available_tools: list[str]) -> str:
        policy = self.harness.stages.get(stage, {})
        order = policy.get('tool_order', [])
        if not set(order) <= set(available_tools):
            raise ValueError('Harness refers to a tool outside the fixed stage permissions')
        sections = []
        if policy:
            sections.append('## Validated reporting harness\n' + '\n'.join(
                f'{name}: {json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value}'
                for name, value in policy.items()))
        key = (_get(state, 'iteration_count', 0), stage)
        if key not in self._selected:
            skill, reason = select_skill(self.skills, stage, state, available_tools)
            self._selected[key] = skill
            self.invocations.append({'stage': stage, 'iteration': key[0], 'skill_id': skill.skill_id if skill else None,
                                     'skill_version': skill.version if skill else None, 'reason': reason})
        skill = self._selected[key]
        if skill:
            lines = [f'## Validated conditional skill: {skill.skill_id}',
                     'Apply these operations only to the applicable claims. Preserve case-specific evidence links.']
            lines.extend(f'{i}. {op["tool"]}: {op["instruction"]}' for i, op in enumerate(skill.operations, 1))
            lines.append('Required evidence checks: ' + '; '.join(skill.evidence_checks))
            sections.append('\n'.join(lines))
        return '\n\n'.join(sections)


def load_runtime(settings: dict) -> RuntimeBundle:
    options = settings.get('evolution', {})
    harness = Harness()
    library = SkillLibrary()
    if options.get('harness_path'):
        harness = Harness.from_dict(json.loads(Path(options['harness_path']).read_text()))
    if options.get('skill_library_path') and options.get('skills_enabled', True):
        library = SkillLibrary.from_dict(json.loads(Path(options['skill_library_path']).read_text()))
    versions = deepcopy(options.get('versions', {}))
    manifest_path = options.get('frozen_manifest_path')
    frozen = None
    if manifest_path:
        from .manifest import verify_freeze
        frozen = verify_freeze(manifest_path)
        expected = frozen.get('versions', {})
        if expected.get('harness') != harness.version or expected.get('skills') != library.version:
            raise ValueError('Runtime harness/skills differ from the frozen configuration')
        versions.update(expected)
    return RuntimeBundle(harness, library, versions, frozen_manifest=frozen)


_REPORTING_MODALITIES = ("", "CXR", "chest_ct", "abdominal_ct", "brain_ct", "brain_mri")


def _source_hashes() -> dict[str, str]:
    """Fingerprint executable templates, tools and context construction code."""
    root = Path(__file__).resolve().parents[1]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*.py"))}


def _effective_agent(config) -> dict:
    return {"max_turns": config.max_turns, "temperature": config.temperature, "max_tokens": config.max_tokens}


def _retrieval_settings(settings: dict) -> dict:
    # Paths may be relocated without changing the corpus; content identity is
    # checked separately against the actual file and frozen snapshot version.
    retrieval = {key: value for key, value in settings.get("retrieval", {}).items() if key != "snapshot_path"}
    return {"retrieval": retrieval, "embedding": deepcopy(settings.get("embedding", {})),
            "hooks": deepcopy(settings.get("hooks", {}))}


def capture_reporting_contract(settings: dict, agent_configs: dict) -> dict:
    """Build a reproducible contract to include in the frozen prompts component.

    ``agent_configs`` are unmodified reporting builders evaluated on empty
    PipelineState objects. Harness and conditional skill instructions are frozen
    independently and are injected only after per-stage verification.
    """
    from src.core.context import ContextBuilder
    from src.core.types import AgentRole, PipelineState
    configs = {key.value if isinstance(key, AgentRole) else key: value for key, value in agent_configs.items()}
    if set(configs) != {role.value for role in AgentRole}:
        raise ValueError("A reporting contract requires all four agent configurations")
    builder = ContextBuilder()
    prompts = {modality: {role.value: builder.build_agent_prompt(
        role, configs[role.value].tool_registry.all_tools(), PipelineState(modality=modality))
        for role in AgentRole} for modality in _REPORTING_MODALITIES}
    return {"schema_version": 1, "source_hashes": _source_hashes(),
            "prompts_by_modality": prompts,
            "agent_parameters": {role: _effective_agent(config) for role, config in configs.items()},
            "tool_schemas": {role: config.tool_registry.get_api_schemas() for role, config in configs.items()},
            "retrieval_and_hooks": _retrieval_settings(settings)}


def _read_component(frozen: dict, name: str) -> dict:
    component = frozen.get("components", {}).get(name)
    if not component:
        raise ValueError(f"Frozen runtime is missing component: {name}")
    path = Path(component["path"])
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != component["sha256"]:
        raise ValueError(f"Frozen component changed: {name}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"Frozen component must be a JSON object: {name}")
    return value


def check_runtime(bundle: RuntimeBundle, settings: dict, llm_client, agent_configs: dict | None = None) -> None:
    """Reject final runs whose actual execution configuration differs from freeze.

    Unfrozen development runs remain supported; PipelineRunner performs its own
    paired-version checks. A declared final manifest always enables these checks,
    including when a caller supplies a RuntimeBundle explicitly.
    """
    from .manifest import verify_freeze
    path = settings.get("evolution", {}).get("frozen_manifest_path")
    frozen = verify_freeze(path) if path else bundle.frozen_manifest
    if frozen is None:
        return
    if not path:
        unhashed = {key: value for key, value in frozen.items() if key != "manifest_hash"}
        if content_hash(unhashed) != frozen.get("manifest_hash"):
            raise ValueError("Explicit frozen runtime manifest hash mismatch")
        # Revalidate component files even for an explicitly passed frozen bundle.
        for name in frozen.get("components", {}):
            component = frozen["components"][name]
            if hashlib.sha256(Path(component["path"]).read_bytes()).hexdigest() != component["sha256"]:
                raise ValueError(f"Frozen component changed: {name}")
    expected = frozen["versions"]
    if expected.get("harness") != bundle.harness.version or expected.get("skills") != bundle.skills.version:
        raise ValueError("Actual harness or skill library differs from the frozen configuration")
    config = getattr(llm_client, "config", None)
    if config is None:
        raise ValueError("Frozen reporting requires an identifiable execution model configuration")
    model = asdict(config) if is_dataclass(config) else dict(vars(config))
    model.pop("api_key", None)
    if content_hash(model) != expected.get("model"):
        raise ValueError("Actual reporting model or decoding configuration differs from freeze")
    transport = getattr(llm_client, "client", None)
    transport_url = getattr(transport, "base_url", None)
    if transport_url is not None and str(transport_url).rstrip("/") != str(model.get("base_url", "")).rstrip("/"):
        raise ValueError("Actual model endpoint differs from the frozen model configuration")
    resources = {key: settings.get(key, {}) for key in ("pipeline", "agents", "limits", "accounting")}
    if content_hash(resources) != expected.get("resource_limits"):
        raise ValueError("Actual reporting resource/accounting settings differ from freeze")
    retrieval = settings.get("retrieval", {})
    if not retrieval.get("require_snapshot") or not retrieval.get("snapshot_path"):
        raise ValueError("Frozen final reporting requires the materialized retrieval snapshot")
    from src.kb.snapshot import CorpusSnapshot
    corpus = CorpusSnapshot(retrieval["snapshot_path"], expected_version=expected["retrieval_snapshot"])
    if retrieval.get("snapshot_version") != corpus.version:
        raise ValueError("Configured retrieval snapshot version differs from freeze")
    prompts = _read_component(frozen, "prompts")
    contract = prompts.get("reporting_contract")
    if not contract or contract.get("schema_version") != 1:
        raise ValueError("Frozen prompts lack the executable reporting contract; regenerate the freeze")
    if contract.get("source_hashes") != _source_hashes():
        raise ValueError("Reporting templates, tools or executable source differ from freeze")
    if contract.get("retrieval_and_hooks") != _retrieval_settings(settings):
        raise ValueError("Actual retrieval, embedding or review-hook settings differ from freeze")
    if _read_component(frozen, "tool_schemas") != contract.get("tool_schemas"):
        raise ValueError("Frozen tool schemas disagree with the reporting contract")
    # A final run may only claim versions that have passed the actual checks.
    bundle.frozen_manifest = frozen
    bundle.frozen_versions = deepcopy(expected)
    if agent_configs is not None:
        from src.core.types import PipelineState
        for config in agent_configs.values():
            verify_agent_config(bundle, settings, config, PipelineState())


def verify_agent_config(bundle: RuntimeBundle, settings: dict, config, state) -> None:
    """Verify actual stage permissions/prompt before adding harness/skill text."""
    if bundle.frozen_manifest is None:
        return
    from src.core.context import ContextBuilder
    from src.core.types import PipelineState
    frozen = bundle.frozen_manifest
    contract = _read_component(frozen, "prompts")["reporting_contract"]
    role = config.role.value
    if config.tool_registry.get_api_schemas() != contract["tool_schemas"].get(role):
        raise ValueError(f"Actual tool permissions or schemas differ from freeze: {role}")
    if _effective_agent(config) != contract["agent_parameters"].get(role):
        raise ValueError(f"Actual agent execution parameters differ from freeze: {role}")
    modality = getattr(state, "modality", "")
    if modality not in contract["prompts_by_modality"]:
        raise ValueError(f"Modality has no frozen reporting prompt: {modality}")
    builder = ContextBuilder()
    tools = config.tool_registry.all_tools()
    static_prompt = builder.build_agent_prompt(config.role, tools, PipelineState(modality=modality))
    if static_prompt != contract["prompts_by_modality"][modality].get(role):
        raise ValueError(f"Actual reporting prompt template differs from freeze: {role}")
    expected_prompt = builder.build_agent_prompt(config.role, tools, state)
    if config.system_prompt != expected_prompt:
        raise ValueError(f"Actual case prompt differs from frozen template plus case context: {role}")
