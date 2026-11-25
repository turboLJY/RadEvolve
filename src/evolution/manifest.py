"""Patient-disjoint development allocations and immutable component manifests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random

from .evaluation import ReferenceFacts
from .schemas import canonical_json, content_hash, strict_keys, write_json


@dataclass(frozen=True)
class DevelopmentCase:
    case_id: str
    patient_id: str  # Canonical source-patient ID, shared across derivative datasets.
    dataset: str
    partition: str
    allocation: str
    image_path: str
    clinical_context: dict
    reference_facts: tuple[dict, ...]

    def __post_init__(self) -> None:
        if self.allocation not in {'evolution', 'development', 'final', 'retrieval'}:
            raise ValueError('Unknown allocation')
        if self.allocation in {'evolution', 'development', 'retrieval'} and self.partition != 'train':
            raise ValueError('Held-in/held-out validation or test cases cannot enter development or retrieval')
        if not self.dataset or not self.case_id or not self.patient_id:
            raise ValueError('Case, dataset and canonical source-patient IDs are required')
        allowed = {'modality', 'body_region', 'body_part', 'study_type', 'view', 'projection',
                   'acquisition', 'sequence', 'sequences', 'sequence_paths', 'contrast_status', 'contrast',
                   'age', 'sex', 'clinical_indication', 'image_path', 'image_paths', 'volume_path', 'nifti_path',
                   'prior_examinations', 'prior_images', 'preview_metadata', 'preview_window', 'preview_plane', 'preview_slice_index', 'image_size', 'series',
                   'orientation', 'spacing', 'slice_index', 'patient_id', 'study_id', 'image_id'}
        if self.clinical_context.keys() - allowed:
            raise ValueError('Only reporting-time clinical and image-access context is permitted')
        def reject_labels(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    name = key.lower()
                    if any(word in name for word in ('ground_truth', 'reference_report', 'reference_fact', 'label', 'outcome', 'disorder')):
                        raise ValueError('Reference reports and benchmark labels are prohibited in generation context')
                    reject_labels(item)
            elif isinstance(value, list):
                for item in value:
                    reject_labels(item)
        reject_labels(self.clinical_context)
        self.reference  # Validate fixed reference schema, including m=0 preflight.

    @property
    def reference(self) -> ReferenceFacts:
        return ReferenceFacts(self.case_id, self.patient_id, self.reference_facts)

    def generation_input(self) -> dict:
        # Reference facts are never forwarded to a reporting runner.
        return {'case_id': self.case_id, 'patient_id': self.patient_id,
                'image_path': self.image_path, 'clinical_context': dict(self.clinical_context)}


@dataclass(frozen=True)
class PatientManifest:
    cases: tuple[DevelopmentCase, ...]
    allocation_seed: int = 20260615

    def __post_init__(self) -> None:
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError('Duplicate case IDs in manifest')
        cohorts = {name: {case.patient_id for case in self.cases if case.allocation == name}
                   for name in ('evolution', 'development', 'final', 'retrieval')}
        for first, second in [('evolution', 'development'), ('evolution', 'final'),
                              ('development', 'final'), ('retrieval', 'final')]:
            overlap = cohorts[first] & cohorts[second]
            if overlap:
                raise ValueError(f'Patient leakage between {first} and {second}: {sorted(overlap)}')

    @classmethod
    def from_dict(cls, value: dict) -> 'PatientManifest':
        strict_keys(value, {'schema_version', 'allocation_seed', 'cases'}, {'schema_version', 'cases'})
        if value['schema_version'] != 1:
            raise ValueError('Unsupported patient manifest schema')
        return cls(tuple(DevelopmentCase(**{**case, 'reference_facts': tuple(case.get('reference_facts', []))})
                         for case in value['cases']), value.get('allocation_seed', 20260615))

    def to_dict(self) -> dict:
        return {'schema_version': 1, 'allocation_seed': self.allocation_seed,
                'cases': [asdict(case) for case in self.cases]}

    def subset(self, name: str, *, evaluable_only: bool = True) -> list[DevelopmentCase]:
        return [case for case in self.cases if case.allocation == name and (case.reference.ids or not evaluable_only)]

    @property
    def unevaluable_case_ids(self) -> list[str]:
        return [case.case_id for case in self.cases if case.allocation in {'evolution', 'development'} and not case.reference.ids]

    def require_sources(self, case_ids: tuple[str, ...], allocation: str = 'evolution') -> None:
        permitted = {case.case_id for case in self.subset(allocation)}
        if not case_ids or not set(case_ids) <= permitted:
            raise ValueError(f'Candidate derivation requires evaluable {allocation} cases only')


def allocate_training_patients(records: list[dict], seed: int = 20260615) -> dict[str, list[str]]:
    """Allocate 10% + 10% per dataset, with source-patient identity consistency.

    Input records must already contain only the intended training partitions.
    Shared patients across datasets cannot silently be assigned contradictory
    roles: such a dataset collection is rejected and requires a joint manifest.
    """
    by_dataset: dict[str, set[str]] = {}
    for row in records:
        if row['partition'] != 'train':
            raise ValueError('Allocation accepts training partitions only')
        by_dataset.setdefault(row['dataset'], set()).add(row['patient_id'])
    roles: dict[str, str] = {}
    result = {'evolution': [], 'development': [], 'remaining': []}
    for dataset, values in sorted(by_dataset.items()):
        patients = sorted(values)
        random.Random(f'{seed}:{dataset}').shuffle(patients)
        count = len(patients) // 10
        if count < 1:
            raise ValueError('At least ten training patients per dataset are required for 10% allocations')
        for index, patient in enumerate(patients):
            role = 'evolution' if index < count else 'development' if index < 2 * count else 'remaining'
            if patient in roles and roles[patient] != role:
                raise ValueError('Shared source patient has conflicting allocations; supply a jointly resolved manifest')
            roles[patient] = role
    for patient, role in sorted(roles.items()):
        result[role].append(patient)
    return result


def freeze_manifest(path: str | Path, components: dict[str, str | Path], versions: dict,
                    evaluation_settings: dict, patient_manifest: PatientManifest) -> dict:
    required = {'model', 'harness', 'skills', 'retrieval_snapshot', 'evaluator', 'resource_limits'}
    if not required <= versions.keys() or any(not versions[key] for key in required):
        raise ValueError('Freeze requires all model, harness, skills, corpus, evaluator and budget versions')
    required_components = {'patient_manifest', 'prompts', 'tool_schemas', 'retrieval_snapshot', 'harness', 'skills', 'evaluation_settings'}
    if not required_components <= components.keys():
        raise ValueError('Freeze requires patient, prompt, tool, corpus, harness, skill and evaluation component files')
    records = {}
    for name, filename in components.items():
        source = Path(filename).resolve()
        if not source.is_file():
            raise ValueError(f'Frozen component is not a file: {source}')
        records[name] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    value = {'schema_version': 1, 'versions': versions, 'components': records,
             'evaluation_settings': evaluation_settings, 'patient_manifest_hash': content_hash(patient_manifest.to_dict()),
             'unevaluable_case_ids': patient_manifest.unevaluable_case_ids}
    value['manifest_hash'] = content_hash(value)
    write_json(path, value)
    return value


def verify_freeze(path: str | Path) -> dict:
    value = json.loads(Path(path).read_text())
    expected = value.pop('manifest_hash')
    if content_hash(value) != expected:
        raise ValueError('Frozen manifest hash mismatch')
    for name, item in value['components'].items():
        source = Path(item['path'])
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Frozen component changed or missing: {name}')
    value['manifest_hash'] = expected
    return value
