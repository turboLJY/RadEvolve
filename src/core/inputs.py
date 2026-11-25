"""Reporting-time context boundary: benchmark targets never enter generation."""
from __future__ import annotations

from copy import deepcopy


EXAM_FIELDS = frozenset({
    "study_type", "view", "modality", "body_part", "body_region", "retrieval_category",
    "age", "sex", "clinical_indication", "indication", "history", "technique", "comparison",
    "acquisition", "projection", "contrast", "contrast_status", "sequence", "sequences", "modal",
    "prior_examinations", "image_paths", "volume_path", "preview_metadata", "image_metadata",
    "preview_window", "dataset", "source_dataset", "split", "sample_index", "sample_id",
    "case_id", "study_id", "patient_id", "volume_name",
})
TARGET_FIELDS = frozenset({
    "ground_truth", "reference", "reference_report", "reference_reports", "reference_facts",
    "findings_section", "impression_section", "findings", "impression", "impressions",
    "positive_labels", "labels", "label", "outcome", "outcomes", "disorders", "benchmark_labels",
})


def _without_targets(value):
    if isinstance(value, dict):
        return {k: _without_targets(v) for k, v in value.items() if k.lower() not in TARGET_FIELDS}
    if isinstance(value, list):
        return [_without_targets(v) for v in value]
    return deepcopy(value)


def reporting_context(exam_info: dict) -> dict:
    """Keep available acquisition/clinical context and source localization only.

    Prior examinations may include their historical report under ``report``;
    the current examination's reference report is never a reporting input.
    Missing demographic/context fields are not filled with invented values.
    """
    if not isinstance(exam_info, dict):
        raise TypeError("exam_info must be a dictionary")
    return {name: _without_targets(value) for name, value in exam_info.items() if name in EXAM_FIELDS}
