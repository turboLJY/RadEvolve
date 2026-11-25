"""Fixed-reference loss and patient-cluster admission gates (Methods 4.3)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
import math
from statistics import mean
from typing import Any

from .schemas import content_hash, strict_keys, string_list

SEVERITY = {'negligible': 0., 'minor': .1, 'major': .5, 'high': .5, 'critical': 1.}
EVIDENCE = {'fully_supported': 0., 'partially_supported': .5, 'contradicted': 1.,
            'irrelevant': 1., 'wrong': 1., 'missing': 1.}


@dataclass(frozen=True)
class ReferenceFacts:
    case_id: str
    patient_id: str
    facts: tuple[dict, ...]

    def __post_init__(self) -> None:
        if not self.case_id or not self.patient_id:
            raise ValueError('Stable case and patient identifiers are required')
        ids = []
        for fact in self.facts:
            strict_keys(fact, {'fact_id', 'text', 'type'}, {'fact_id', 'text'})
            if not isinstance(fact['fact_id'], str) or not fact['fact_id'] or not fact['text']:
                raise ValueError('Reference facts require stable IDs and clinical text')
            ids.append(fact['fact_id'])
        if len(ids) != len(set(ids)):
            raise ValueError('Reference facts must be deduplicated before evaluation')

    @property
    def ids(self) -> set[str]:
        return {fact['fact_id'] for fact in self.facts}

    @property
    def version(self) -> str:
        return content_hash(asdict(self))


@dataclass(frozen=True)
class Assessment:
    """Semantic IDs are assigned by the fixed evaluator across both paired arms.

    error_id deduplicates equivalent errors; claim_id deduplicates equivalent
    propositions; reference IDs come exclusively from frozen ReferenceFacts.
    Unverifiable claims cannot enter correct_claim_ids or represented facts.
    """
    reference_version: str
    clinical_errors: tuple[dict, ...]
    evidence_relations: tuple[dict, ...]
    covered_fact_ids: frozenset[str]
    correct_claim_ids: frozenset[str]
    unverifiable_claim_ids: frozenset[str]
    target_defect_present: bool
    target_required_fact_ids: frozenset[str]
    unresolved_severe: bool = False
    adjudication_complete: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> 'Assessment':
        fields = set(cls.__dataclass_fields__)
        strict_keys(data, fields, fields)
        raw = dict(data)
        for name in ('covered_fact_ids', 'correct_claim_ids', 'unverifiable_claim_ids', 'target_required_fact_ids'):
            raw[name] = frozenset(string_list(raw[name], name))
        for name in ('target_defect_present', 'unresolved_severe', 'adjudication_complete'):
            if type(raw[name]) is not bool:
                raise ValueError(f'{name} must be boolean')
        for error in raw['clinical_errors']:
            strict_keys(error, {'error_id', 'severity', 'kind', 'confirmed'}, {'error_id', 'severity', 'kind', 'confirmed'})
            if not error['error_id'] or error['severity'] not in SEVERITY or type(error['confirmed']) is not bool:
                raise ValueError('Invalid clinical error record')
        for relation in raw['evidence_relations']:
            strict_keys(relation, {'claim_id', 'semantic_id', 'relation'}, {'claim_id', 'relation'})
            if not relation['claim_id'] or relation['relation'] not in EVIDENCE:
                raise ValueError('Invalid claim-evidence relation')
        raw['clinical_errors'] = tuple(raw['clinical_errors'])
        raw['evidence_relations'] = tuple(raw['evidence_relations'])
        result = cls(**raw)
        if result.correct_claim_ids & result.unverifiable_claim_ids:
            raise ValueError('An unverifiable claim cannot be correct')
        return result

    def to_dict(self) -> dict:
        data = asdict(self)
        for name in ('covered_fact_ids', 'correct_claim_ids', 'unverifiable_claim_ids', 'target_required_fact_ids'):
            data[name] = sorted(data[name])
        data['clinical_errors'] = list(data['clinical_errors'])
        data['evidence_relations'] = list(data['evidence_relations'])
        return data

    def validate_reference(self, reference: ReferenceFacts) -> None:
        if self.reference_version != reference.version:
            raise ValueError('Reference facts changed during candidate evaluation')
        if not self.covered_fact_ids <= reference.ids or not self.target_required_fact_ids <= reference.ids:
            raise ValueError('Assessment contains unknown reference fact identifiers')

    def errors(self) -> dict[str, float]:
        errors: dict[str, float] = {}
        for error in self.clinical_errors:
            if error['confirmed'] and error['kind'] != 'omission':
                errors[error['error_id']] = max(errors.get(error['error_id'], 0.), SEVERITY[error['severity']])
        return errors

    def loss(self, reference: ReferenceFacts) -> dict[str, float]:
        self.validate_reference(reference)
        m = len(reference.ids)
        if not m:
            raise ValueError('Case has no required reference facts and is unevaluable')
        penalties: dict[str, float] = {}
        for relation in self.evidence_relations:
            cid = relation.get('semantic_id') or relation['claim_id']
            penalties[cid] = max(penalties.get(cid, 0.), EVIDENCE[relation['relation']])
        clinical = sum(self.errors().values()) / m
        evidence = sum(penalties.values()) / m
        omission = len(reference.ids - self.covered_fact_ids) / m
        return {'clinical': clinical, 'evidence': evidence, 'omission': omission,
                'total': .50 * clinical + .25 * evidence + .25 * omission}

    @property
    def target_occurs(self) -> bool:
        # Deleting the required fact never counts as a successful repair.
        return self.target_defect_present or not self.target_required_fact_ids <= self.covered_fact_ids


@dataclass(frozen=True)
class RunOutput:
    report: dict
    complete: bool
    cost_cny: float | None
    within_resource_limits: bool
    trace: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.cost_cny is not None and (not math.isfinite(self.cost_cny) or self.cost_cny < 0):
            raise ValueError('Cost must be a finite nonnegative CNY value or missing')


@dataclass(frozen=True)
class PairedEvaluation:
    reference: ReferenceFacts
    repetition: int
    control: Assessment
    candidate: Assessment
    control_run: RunOutput
    candidate_run: RunOutput
    retained_control_claim_ids: frozenset[str]

    def __post_init__(self) -> None:
        self.control.validate_reference(self.reference)
        self.candidate.validate_reference(self.reference)
        if self.control.target_required_fact_ids != self.candidate.target_required_fact_ids:
            raise ValueError('Target correction requirements must be fixed between arms')
        if not self.retained_control_claim_ids <= self.control.correct_claim_ids:
            raise ValueError('Retention contains claims not judged correct in the control')
        if self.repetition < 0:
            raise ValueError('Repetition must be nonnegative')

    @property
    def new_severe_errors(self) -> set[str]:
        original = self.control.errors()
        return {key for key, value in self.candidate.errors().items()
                if value >= .5 and (original.get(key, 0.) < .5 or value > original.get(key, 0.))}

    @property
    def valid(self) -> bool:
        return (self.control_run.complete and self.candidate_run.complete
                and self.control_run.within_resource_limits and self.candidate_run.within_resource_limits
                and self.control.adjudication_complete and self.candidate.adjudication_complete
                and not self.control.unresolved_severe and not self.candidate.unresolved_severe
                and bool(self.reference.ids))

    def metrics(self) -> dict[str, float | None]:
        correct = self.control.correct_claim_ids
        return {
            'gain': self.control.loss(self.reference)['total'] - self.candidate.loss(self.reference)['total'],
            'retention': len(self.retained_control_claim_ids) / len(correct) if correct else None,
            'coverage_difference': (len(self.candidate.covered_fact_ids) - len(self.control.covered_fact_ids)) / len(self.reference.ids),
            'candidate_cost': self.candidate_run.cost_cny,
        }

    def to_dict(self) -> dict:
        return {'reference': asdict(self.reference), 'repetition': self.repetition,
                'control': self.control.to_dict(), 'candidate': self.candidate.to_dict(),
                'control_run': asdict(self.control_run), 'candidate_run': asdict(self.candidate_run),
                'retained_control_claim_ids': sorted(self.retained_control_claim_ids)}

    @classmethod
    def from_dict(cls, data: dict) -> 'PairedEvaluation':
        fields = set(cls.__dataclass_fields__)
        strict_keys(data, fields, fields)
        reference = ReferenceFacts(**{**data['reference'], 'facts': tuple(data['reference']['facts'])})
        return cls(reference, data['repetition'], Assessment.from_dict(data['control']),
                   Assessment.from_dict(data['candidate']), RunOutput(**data['control_run']),
                   RunOutput(**data['candidate_run']), frozenset(string_list(data['retained_control_claim_ids'], 'retained_control_claim_ids')))


@dataclass(frozen=True)
class AcceptanceSettings:
    local_repetitions: int = 5
    validation_repetitions: int = 3
    bootstrap_resamples: int = 20000
    bootstrap_seed: int = 20260615
    confidence: float = .95
    min_gain: float = .01
    min_retention: float = .99
    min_coverage_difference: float = -.01
    cost_multiplier: float = 1.25

    def __post_init__(self) -> None:
        if min(self.local_repetitions, self.validation_repetitions, self.bootstrap_resamples) <= 0:
            raise ValueError('Repetition counts must be positive')
        if not 0 < self.confidence < 1 or self.min_gain <= 0 or not 0 <= self.min_retention <= 1 or self.cost_multiplier <= 0:
            raise ValueError('Invalid acceptance thresholds')


@dataclass(frozen=True)
class Decision:
    accepted: bool
    reasons: tuple[str, ...]
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {'accepted': self.accepted, 'reasons': list(self.reasons), 'metrics': self.metrics}


def _integrity(pairs: list[PairedEvaluation], expected_cases: set[str], repetitions: int) -> list[str]:
    reasons = []
    by_case: dict[str, list[PairedEvaluation]] = defaultdict(list)
    for pair in pairs:
        by_case[pair.reference.case_id].append(pair)
    if set(by_case) != expected_cases:
        reasons.append('missing_or_unexpected_cases')
    for case_id, records in by_case.items():
        if len(records) != repetitions or {p.repetition for p in records} != set(range(repetitions)):
            reasons.append(f'incomplete_repetitions:{case_id}')
        if len({p.control.target_required_fact_ids for p in records}) != 1:
            reasons.append(f'changed_target_requirements:{case_id}')
        if len({p.reference.version for p in records}) != 1:
            reasons.append(f'changed_reference:{case_id}')
    if not pairs or any(not p.valid for p in pairs):
        reasons.append('incomplete_or_unadjudicated_runs')
    if any(p.new_severe_errors for p in pairs):
        reasons.append('new_critical_or_high_severity_error')
    return reasons


def local_screen(pairs: list[PairedEvaluation], case_id: str,
                 settings: AcceptanceSettings = AcceptanceSettings()) -> Decision:
    reasons = _integrity(pairs, {case_id}, settings.local_repetitions)
    if reasons:
        return Decision(False, tuple(reasons))
    gain = mean(p.metrics()['gain'] for p in pairs)
    control = sum(p.control.target_occurs for p in pairs)
    candidate = sum(p.candidate.target_occurs for p in pairs)
    if gain <= 0:
        reasons.append('nonpositive_loss_reduction')
    if candidate >= control:
        reasons.append('target_defect_not_reduced')
    return Decision(not reasons, tuple(reasons), {'mean_loss_reduction': gain, 'control_target_count': control, 'candidate_target_count': candidate})


def patient_metrics(pairs: list[PairedEvaluation]) -> dict[str, dict[str, float | None]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for pair in pairs:
        grouped[pair.reference.patient_id].append(pair.metrics())
    return {patient: {key: mean(values) if (values := [r[key] for r in rows if r[key] is not None]) else None
                      for key in ('gain', 'retention', 'coverage_difference', 'candidate_cost')}
            for patient, rows in sorted(grouped.items())}


def acceptance(pairs: list[PairedEvaluation], expected_cases: set[str], reference_cost_cny: float,
               settings: AcceptanceSettings = AcceptanceSettings(), *, require_target_reduction: bool = True) -> Decision:
    """Average repetitions within patients; bootstrap matched patient clusters.

    Missing retention is excluded only from retention. Missing cost, incomplete
    cases, unresolved severe issues or an unestimable bound prohibit adoption.
    """
    if not math.isfinite(reference_cost_cny) or reference_cost_cny <= 0:
        raise ValueError('Initial-configuration reference cost must be measured before selection')
    reasons = _integrity(pairs, expected_cases, settings.validation_repetitions)
    if any(p.candidate_run.cost_cny is None or p.control_run.cost_cny is None for p in pairs):
        reasons.append('missing_complete_run_cost')
    if reasons:
        return Decision(False, tuple(reasons))
    control_target = sum(p.control.target_occurs for p in pairs)
    candidate_target = sum(p.candidate.target_occurs for p in pairs)
    if require_target_reduction and candidate_target >= control_target:
        reasons.append('target_defect_not_reduced')
    rows = patient_metrics(pairs)
    import numpy as np
    keys = ('gain', 'retention', 'coverage_difference', 'candidate_cost')
    array = np.asarray([[float('nan') if row[key] is None else row[key] for key in keys] for row in rows.values()])
    if not len(array) or np.isnan(array).all(axis=0).any():
        return Decision(False, tuple(reasons + ['unestimable_acceptance_metric']))
    rng = np.random.default_rng(settings.bootstrap_seed)
    # Chunking avoids an O(resamples * cohort size * metrics) allocation.
    samples = []
    for start in range(0, settings.bootstrap_resamples, 256):
        indices = rng.integers(0, len(array), size=(min(256, settings.bootstrap_resamples - start), len(array)))
        selected = array[indices]
        counts = np.sum(~np.isnan(selected), axis=1)
        sums = np.nansum(selected, axis=1)
        samples.append(np.divide(sums, counts, out=np.full(sums.shape, np.nan), where=counts > 0))
    draws = np.concatenate(samples)
    # A patient can legitimately have no correct control claims. Such a patient
    # contributes to all other metrics; retention alone omits that missing value.
    # Draws containing no estimable retention are omitted only from that bound.
    if np.isnan(draws).all(axis=0).any():
        return Decision(False, tuple(reasons + ['unestimable_bootstrap_bound']))
    bounds = np.nanquantile(draws, 1 - settings.confidence, axis=0)
    means = np.nanmean(array, axis=0)
    thresholds = (settings.min_gain, settings.min_retention, settings.min_coverage_difference)
    for i, name in enumerate(keys[:3]):
        if bounds[i] < thresholds[i] - 1e-12:
            reasons.append(f'{name}_lower_bound_failed')
    if means[3] > settings.cost_multiplier * reference_cost_cny + 1e-12:
        reasons.append('cost_ceiling_exceeded')
    metrics = {'patients': len(rows), 'paired_runs': len(pairs),
               'means': dict(zip(keys, map(float, means))),
               'lower_bounds': dict(zip(keys[:3], map(float, bounds[:3]))),
               'control_target_count': control_target, 'candidate_target_count': candidate_target,
               'reference_cost_cny': reference_cost_cny, 'cost_ceiling_cny': settings.cost_multiplier * reference_cost_cny,
               'bootstrap': asdict(settings),
               'estimable_bootstrap_resamples': dict(zip(keys, map(int, np.sum(~np.isnan(draws), axis=0))))}
    return Decision(not reasons, tuple(reasons), metrics)
