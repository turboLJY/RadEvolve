"""Method regressions: source images, observation semantics, and frozen retrieval."""
from __future__ import annotations

import io
import json
import base64

import numpy as np
import pytest
from PIL import Image

from src.core.tool import ToolUseContext
from src.core.types import PipelineState
from src.data.image_store import SessionImageStore, ImageViewLimitError
from src.data.chest_ct_jsonl_loader import _window_slice, load_nifti_preview
from src.kb.snapshot import CorpusSnapshot
from src.tools.evidence.standardize_obs import StandardizeObservationTool
from src.tools.evidence.literature_search import LiteratureSearchTool
from src.tools.evidence.case_retrieval import CaseRetrievalTool
from src.tools.visual.select_slice import SelectSliceTool
from src.tools.visual.link_evidence import LinkEvidenceTool


def make_context(**kwargs):
    state = PipelineState(session_id="method", **kwargs)
    return ToolUseContext(session_id="method", extra={"pipeline_state": state, "image_store": SessionImageStore("method")})


def test_distinct_view_budget_and_duplicate_reuse():
    store = SessionImageStore("case", max_views=2)
    image = Image.new("RGB", (20, 20))
    first = store.save(image, "slice 1", metadata={"slice_index": 1})
    assert store.save(image, "same slice", metadata={"slice_index": 1}) == first
    store.save(image, "different origin", metadata={"slice_index": 2})
    with pytest.raises(ImageViewLimitError):
        store.save(image, "third slice", metadata={"slice_index": 3})
    assert store.count() == 2


def test_derived_view_keeps_origin_and_image_transport_size():
    store = SessionImageStore("case")
    image = Image.new("RGB", (2048, 1024))
    first = store.save(image, "source", "select_slice", {"sequence_id": "T2", "slice_index": 7, "transforms": [{"operation": "rotate90"}]})
    second = store.save_derived(image.crop((0, 0, 512, 512)), "ROI", "crop_roi", first, {"operation": "crop", "crop_box": {"x": 0, "y": 0, "width": 512, "height": 512}})
    metadata = store.get_metadata(second)
    assert metadata["sequence_id"] == "T2" and metadata["slice_index"] == 7
    assert [t["operation"] for t in metadata["transforms"]] == ["rotate90", "crop"]
    assert store.get_tool_chain(second) == ["select_slice", "crop_roi"]
    transmitted = Image.open(io.BytesIO(base64.b64decode(store.to_base64(first))))
    assert transmitted.size == (1024, 512)


def test_constant_ct_still_uses_physical_window():
    arr = np.full((3, 3), 50.0)
    image = np.asarray(_window_slice(arr, "soft_tissue"))
    assert np.all(image == 127)  # 50 HU maps to the middle of 50/350 HU.


def make_volume(tmp_path):
    import nibabel as nib
    data = np.arange(4 * 5 * 9, dtype=np.float32).reshape(4, 5, 9)
    path = tmp_path / "scan.nii.gz"
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


def test_volume_preview_records_real_axes_affine_and_rotation(tmp_path):
    path = make_volume(tmp_path)
    image, meta = load_nifti_preview(path, window="mri")
    assert image.size == (4, 5)
    assert meta["array_axis"] == 2 and meta["slice_index"] == 4
    assert meta["orientation"] == ["R", "A", "S"]
    assert meta["transforms"] == [{"operation": "rotate90", "k": 1}]
    with pytest.raises(IndexError):
        load_nifti_preview(path, slice_index=9)


@pytest.mark.asyncio
async def test_volume_only_supplied_sequence_and_coarse_before_targeted(tmp_path):
    path = make_volume(tmp_path)
    ctx = make_context(modality="brain_mri", exam_info={"image_paths": [str(path)]})
    tool = SelectSliceTool()
    assert (await tool.call({"sequence_id": "not_this_exam"}, ctx)).is_error
    assert (await tool.call({"strategy": "targeted", "slice_index": 5}, ctx)).is_error
    coarse = await tool.call({"strategy": "coarse"}, ctx)
    assert not coarse.is_error
    assert len(coarse.metadata["image_ids"]) == 5
    target = await tool.call({"strategy": "targeted", "slice_index": 5}, ctx)
    assert not target.is_error
    assert ctx.extra["pipeline_state"].exam_info["image_metadata"]["slice_index"] == 5
    assert (await tool.call({"strategy": "targeted", "slice_index": 999}, ctx)).is_error


@pytest.mark.asyncio
async def test_normalization_preserves_visual_id_negation_laterality_and_source():
    ctx = make_context()
    image_id = ctx.extra["image_store"].save(Image.new("RGB", (20, 20)), "source")
    linked = await LinkEvidenceTool().call({"finding_description": "No left pleural effusion", "image_evidence_ids": [image_id], "confidence": 0.9, "anatomy": "pleura"}, ctx)
    oid = json.loads(linked.data)["observation_id"]
    result = await StandardizeObservationTool().call({"observation_text": "No left pleural effusion", "observation_id": oid}, ctx)
    data = json.loads(result.data)
    assert data["observation_id"] == oid
    assert data["polarity"] == "negative" and data["laterality"] == "left"
    assert data["image_evidence_ids"] == [image_id]
    assert len(ctx.extra["pipeline_state"].observation_graph) == 1
    assert oid in ctx.extra["pipeline_state"].dependency_graph.descendants({image_id})


def snapshot(tmp_path):
    return CorpusSnapshot.materialize(tmp_path / "corpus.json", [
        {"document_id": "GL1", "source_type": "guideline", "content": "Pleural effusion guidance", "category": "chest_xray"},
        {"document_id": "GL2", "source_type": "guideline", "content": "Cardiomegaly guidance", "category": "chest_xray"},
        {"document_id": "CASE1", "source_type": "case", "content": "Pleural effusion", "patient_id": "train1", "partition": "train"},
    ], ["guideline", "case", "radlex", "snomed"])


def test_snapshot_allows_changed_queries_but_detects_mutation_and_leakage(tmp_path):
    corpus = snapshot(tmp_path)
    assert corpus.search("effusion", ["guideline"])[0]["document_id"] == "GL1"
    assert corpus.search("cardiomegaly", ["guideline"])[0]["document_id"] == "GL2"
    with pytest.raises(ValueError, match="declared"):
        corpus.search("effusion", ["pubmed"])
    path = tmp_path / "corpus.json"
    payload = json.loads(path.read_text())
    payload["records"][0]["content"] = "changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        CorpusSnapshot(path)
    with pytest.raises(ValueError, match="prohibited"):
        CorpusSnapshot.materialize(tmp_path / "leak.json", [{"document_id": "case", "source_type": "case", "content": "x", "patient_id": "test1", "partition": "test"}], ["case"])


@pytest.mark.asyncio
async def test_frozen_retrieval_avoids_network_and_links_observations(tmp_path, monkeypatch):
    corpus = snapshot(tmp_path)
    ctx = make_context(modality="CXR")
    obs_result = await StandardizeObservationTool().call({"observation_text": "Pleural effusion"}, ctx)
    oid = json.loads(obs_result.data)["observation_id"]
    def no_network(*args, **kwargs):
        raise AssertionError("Frozen retrieval attempted network access")
    monkeypatch.setattr("httpx.AsyncClient", no_network)
    tool = LiteratureSearchTool(remote_base_url="https://unused.invalid", snapshot=corpus)
    args = {"query": "effusion", "observation_ids": [oid]}
    one = await tool.call(args, ctx)
    two = await tool.call(args, ctx)
    assert not one.is_error and not two.is_error
    state = ctx.extra["pipeline_state"]
    assert len(state.retrieved_evidence) == 1
    ev = state.retrieved_evidence[0]
    assert ev.metadata["snapshot_version"] == corpus.version
    assert ev.metadata["observation_ids"] == [oid]
    assert ev.evidence_id in state.dependency_graph.descendants({oid})
    cases = await CaseRetrievalTool(snapshot=corpus).call({"observations": ["effusion"], "observation_ids": [oid]}, ctx)
    assert not cases.is_error
    bad = await tool.call({"query": "effusion", "observation_ids": ["UNKNOWN"]}, ctx)
    assert bad.is_error


@pytest.mark.asyncio
async def test_visual_repair_replaces_dependencies_and_invalidates_only_descendants():
    ctx = make_context()
    state, store = ctx.extra["pipeline_state"], ctx.extra["image_store"]
    image1 = store.save(Image.new("RGB", (20, 20)), "first", metadata={"slice_index": 1})
    image2 = store.save(Image.new("RGB", (20, 20)), "second", metadata={"slice_index": 2})
    tool = LinkEvidenceTool()
    args = {"finding_description": "Left opacity", "image_evidence_ids": [image1], "confidence": 0.8}
    result = await tool.call(args, ctx)
    oid = json.loads(result.data)["observation_id"]
    graph = state.dependency_graph
    graph.add_node("F1", "finding", {})
    graph.add_node("F2", "finding", {})
    graph.add_edge(oid, "F1")
    state.reviewed_claim_ids = {"F1", "F2"}
    state.claim_reviews = {"F1": {"pass": True}, "F2": {"pass": True}}
    state.iteration_count = 2
    rejected = await tool.call({**args, "observation_id": oid, "finding_description": "Revised left opacity"}, ctx)
    assert rejected.is_error
    state.affected_node_ids = {oid}
    updated = await tool.call({**args, "observation_id": oid, "finding_description": "Revised left opacity", "image_evidence_ids": [image2]}, ctx)
    assert not updated.is_error
    assert (image1, oid) not in graph.edges and (image2, oid) in graph.edges
    assert "F1" in state.affected_node_ids and "F2" not in state.affected_node_ids
    assert state.reviewed_claim_ids == {"F2"} and "F1" not in state.claim_reviews
