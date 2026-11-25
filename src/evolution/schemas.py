"""Versioned, strict records for the offline procedure in Methods 4.2–4.3.

No model weights, tool implementation, permission, evaluator or budget can be
changed by a harness proposal. JSON serialisation is deliberately explicit.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any

STAGES = ('visual_analysis', 'evidence_retrieval', 'report_writing', 'quality_review')
EDITABLE_FIELDS = frozenset({
    'task_instructions', 'tool_order', 'retrieval_query_rules',
    'evidence_selection_rules', 'claim_binding_rules', 'remediation_routing',
})
SCOPE_KEYS = frozenset({'modality', 'anatomy', 'claim_type', 'defect_type', 'evidence_availability'})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def strict_keys(value: dict, allowed: set | frozenset, required: set | frozenset = frozenset()) -> None:
    if not isinstance(value, dict):
        raise ValueError('A JSON object is required')
    extra = value.keys() - allowed
    missing = required - value.keys()
    if extra or missing:
        raise ValueError(f'Invalid fields: extra={sorted(extra)}, missing={sorted(missing)}')


def string_list(value: Any, name: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value):
        raise ValueError(f'{name} must be a list of nonempty strings')
    if nonempty and not value:
        raise ValueError(f'{name} must not be empty')
    if len(set(value)) != len(value):
        raise ValueError(f'{name} contains duplicates')
    return list(value)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@dataclass(frozen=True)
class HarnessEdit:
    edit_id: str
    stage: str
    field: str
    value: Any
    target_defect: str
    assessment_criteria: str
    preserve: tuple[str, ...]
    source_case_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict) -> 'HarnessEdit':
        fields = set(cls.__dataclass_fields__)
        strict_keys(data, fields, fields)
        for name in ('edit_id', 'target_defect', 'assessment_criteria'):
            if not isinstance(data[name], str) or not data[name].strip():
                raise ValueError(f'{name} is required')
        if data['stage'] not in STAGES or data['field'] not in EDITABLE_FIELDS:
            raise ValueError('Harness proposal changes a prohibited stage or field')
        value = data['value']
        if data['field'] == 'tool_order':
            string_list(value, 'tool_order', nonempty=True)
        elif not isinstance(value, str) or not value.strip():
            raise ValueError('An instruction edit must contain nonempty text')
        return cls(**{**deepcopy(data),
                      'preserve': tuple(string_list(data['preserve'], 'preserve', nonempty=True)),
                      'source_case_ids': tuple(string_list(data['source_case_ids'], 'source_case_ids', nonempty=True))})

    def to_dict(self) -> dict:
        value = asdict(self)
        value['preserve'] = list(self.preserve)
        value['source_case_ids'] = list(self.source_case_ids)
        return value


@dataclass(frozen=True)
class Harness:
    stages: dict[str, dict] = field(default_factory=lambda: {stage: {} for stage in STAGES})
    parent_version: str | None = None
    accepted_edits: tuple[str, ...] = ()

    @property
    def version(self) -> str:
        return 'h-' + content_hash(self.to_dict(include_version=False))[:16]

    def to_dict(self, include_version: bool = True) -> dict:
        result = {'schema_version': 1, 'stages': deepcopy(self.stages),
                  'parent_version': self.parent_version, 'accepted_edits': list(self.accepted_edits)}
        if include_version:
            result['version'] = self.version
        return result

    @classmethod
    def from_dict(cls, value: dict) -> 'Harness':
        strict_keys(value, {'schema_version', 'stages', 'parent_version', 'accepted_edits', 'version'}, {'schema_version', 'stages'})
        if value['schema_version'] != 1:
            raise ValueError('Unsupported harness schema')
        strict_keys(value['stages'], set(STAGES))
        stages = {stage: deepcopy(value['stages'].get(stage, {})) for stage in STAGES}
        for stage, policy in stages.items():
            strict_keys(policy, EDITABLE_FIELDS)
            for key, item in policy.items():
                if key == 'tool_order':
                    string_list(item, key, nonempty=True)
                elif not isinstance(item, str) or not item.strip():
                    raise ValueError(f'{stage}.{key} must be nonempty text')
        result = cls(stages, value.get('parent_version'), tuple(string_list(value.get('accepted_edits', []), 'accepted_edits')))
        if 'version' in value and value['version'] != result.version:
            raise ValueError('Harness content hash mismatch')
        return result

    def apply(self, edit: HarnessEdit, permitted_tools: dict[str, list[str]]) -> 'Harness':
        # Revalidate even for a dataclass constructed directly by a caller.
        edit = HarnessEdit.from_dict(edit.to_dict())
        if edit.field == 'tool_order' and not set(edit.value).issubset(permitted_tools.get(edit.stage, [])):
            raise ValueError('Harness tool_order includes a tool outside existing permissions')
        if edit.edit_id in self.accepted_edits:
            raise ValueError('Edit already applied')
        stages = deepcopy(self.stages)
        stages[edit.stage][edit.field] = deepcopy(edit.value)
        return Harness(stages, self.version, self.accepted_edits + (edit.edit_id,))


@dataclass(frozen=True)
class Skill:
    skill_id: str
    stage: str
    applicability: dict[str, list[str]]
    required_inputs: tuple[str, ...]
    operations: tuple[dict, ...]
    evidence_checks: tuple[str, ...]
    exclusions: tuple[dict, ...]
    priority: int
    source_edit_id: str
    source_case_ids: tuple[str, ...]
    status: str = 'candidate'
    validation_record: dict = field(default_factory=dict)
    tested_scope: dict = field(default_factory=dict)
    deactivation_reason: str = ''

    @property
    def version(self) -> str:
        return 's-' + content_hash(self.to_dict(include_version=False))[:16]

    def to_dict(self, include_version: bool = True) -> dict:
        result = asdict(self)
        for name in ('required_inputs', 'operations', 'evidence_checks', 'exclusions', 'source_case_ids'):
            result[name] = list(result[name])
        if include_version:
            result['version'] = self.version
        return result

    @classmethod
    def from_dict(cls, data: dict) -> 'Skill':
        fields = set(cls.__dataclass_fields__)
        required = fields - {'status', 'validation_record', 'tested_scope', 'deactivation_reason'}
        strict_keys(data, fields | {'version'}, required)
        raw = deepcopy(data)
        expected_version = raw.pop('version', None)
        if raw['stage'] not in STAGES:
            raise ValueError('Unknown skill stage')
        for name in ('skill_id', 'source_edit_id'):
            if not isinstance(raw[name], str) or not raw[name].strip():
                raise ValueError(f'{name} is required')
        for scope in [raw['applicability'], *raw['exclusions']]:
            strict_keys(scope, SCOPE_KEYS)
            if not scope:
                raise ValueError('Empty applicability/exclusion condition is prohibited')
            for name, values in scope.items():
                string_list(values, name, nonempty=True)
        for name in ('required_inputs', 'evidence_checks', 'source_case_ids'):
            raw[name] = tuple(string_list(raw[name], name, nonempty=True))
        if not isinstance(raw['operations'], list) or not raw['operations']:
            raise ValueError('Skill requires ordered operations')
        for operation in raw['operations']:
            strict_keys(operation, {'tool', 'instruction'}, {'tool', 'instruction'})
            if any(not isinstance(value, str) or not value.strip() for value in operation.values()):
                raise ValueError('Skill operations require tool and instruction text')
        if not isinstance(raw['priority'], int) or isinstance(raw['priority'], bool):
            raise ValueError('Skill priority must be an integer')
        if raw.get('status', 'candidate') not in {'candidate', 'active', 'rejected', 'disabled'}:
            raise ValueError('Invalid skill lifecycle status')
        raw['operations'] = tuple(raw['operations'])
        raw['exclusions'] = tuple(raw['exclusions'])
        result = cls(**raw)
        if expected_version is not None and expected_version != result.version:
            raise ValueError('Skill content hash mismatch')
        if result.status == 'active' and (not result.validation_record or not result.tested_scope):
            raise ValueError('Active skills require validation evidence and tested scope')
        return result


@dataclass(frozen=True)
class SkillLibrary:
    skills: tuple[Skill, ...] = ()

    @property
    def version(self) -> str:
        return 'S-' + content_hash([s.to_dict() for s in self.skills])[:16]

    def to_dict(self) -> dict:
        return {'schema_version': 1, 'version': self.version, 'skills': [s.to_dict() for s in self.skills]}

    @classmethod
    def from_dict(cls, data: dict) -> 'SkillLibrary':
        strict_keys(data, {'schema_version', 'version', 'skills'}, {'schema_version', 'skills'})
        if data['schema_version'] != 1 or not isinstance(data['skills'], list):
            raise ValueError('Invalid skill-library schema')
        skills = tuple(Skill.from_dict(s) for s in data['skills'])
        if len({s.skill_id for s in skills}) != len(skills):
            raise ValueError('Duplicate skill identifiers')
        result = cls(skills)
        if 'version' in data and data['version'] != result.version:
            raise ValueError('Skill-library content hash mismatch')
        return result

    def with_skill(self, skill: Skill) -> 'SkillLibrary':
        return SkillLibrary(tuple(s for s in self.skills if s.skill_id != skill.skill_id) + (skill,))

    def without(self, skill_id: str) -> 'SkillLibrary':
        return SkillLibrary(tuple(s for s in self.skills if s.skill_id != skill_id))


@dataclass(frozen=True)
class Versions:
    model: str
    harness: str
    skills: str
    retrieval_snapshot: str
    evaluator: str
    resource_limits: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Checkpoint:
    case_id: str
    before_stage: str
    versions: Versions
    conversation_history: list
    working_memory: dict
    observations: list
    evidence_records: list
    dependency_graph: dict
    completed_tool_outputs: list
    state: dict

    def __post_init__(self) -> None:
        if self.before_stage not in STAGES:
            raise ValueError('Checkpoint requires a reporting stage')
        # A JSON round trip rejects unserializable hidden state and nonfinite numbers.
        canonical_json(self.to_dict())

    def compatible(self, versions: Versions) -> bool:
        return self.versions == versions

    def clone(self) -> 'Checkpoint':
        return Checkpoint.from_dict(json.loads(canonical_json(self.to_dict())))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> 'Checkpoint':
        fields = set(cls.__dataclass_fields__)
        strict_keys(value, fields, fields)
        raw = deepcopy(value)
        strict_keys(raw['versions'], set(Versions.__dataclass_fields__), set(Versions.__dataclass_fields__))
        raw['versions'] = Versions(**raw['versions'])
        return cls(**raw)
