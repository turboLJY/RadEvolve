# Image and retrieval implementation aligned with Methods

This note describes the image/evidence code changes corresponding to RadEvolve Methods 4.1-4.3. It is implementation documentation, not an experimental result or a replacement manuscript.

## Current-examination images

`SessionImageStore(session_id, max_views=32)` owns one shared view budget for the whole case, including all sequences and remediation passes. Reusing the same pixels with the same source/transform metadata reuses an identifier. Identical-looking slices with different source slice or sequence identifiers remain distinct views. A new view beyond the budget raises `ImageViewLimitError`; it does not replace an existing view.

The volumetric JSONL loaders use the central slice of array axis 2 by default, designate it axial and rotate the display by 90 degrees. This is an array-axis convention rather than anatomical reorientation. Saved metadata includes the NIfTI affine, source orientation codes, array axis, slice index, original volume shape and selected frame. For a volume with an extra dimension the default frame is index 0. All sequence volumes to inspect must be explicitly supplied with the current examination.

CT previews use centre/width -600/1500 HU for chest images and 50/350 HU for abdominal images. A constant-intensity CT image retains physical HU windowing. MRI uses the first and ninety-ninth intensity percentiles. Derived view transport is PNG with Lanczos downsampling to at most 1,024 pixels on the longest dimension; the stored source image remains at source resolution.

`select_slice_or_view` keeps the radiograph `view_type` interface and additionally accepts:

```json
{"sequence_id": "sequence_0", "strategy": "coarse", "plane": "axial", "window": "lung"}
```

Coarse inspection samples the first, quarter, middle, three-quarter and last array positions, with duplicates removed for small volumes. A subsequent targeted call can inspect a selected neighbouring slice:

```json
{"sequence_id": "sequence_0", "strategy": "targeted", "plane": "axial", "slice_index": 12, "window": "lung"}
```

Targeted inspection requires completed coarse inspection of that sequence and plane. Arbitrary filesystem paths are not accepted by the tool. Available sequence IDs are derived from the examination's `image_paths`/`volume_path` and `sequences`; out-of-range slices and sequences outside the supplied examination fail explicitly.

The selected slice becomes `PipelineState.active_image_id`. `crop_roi` accepts an optional `image_id` and otherwise crops that active view. Crop, zoom, annotation, probe and composite tools preserve their parent IDs and transformations. For a native CT slice, `multi_window_render` reloads the underlying volume values, applies the requested HU window and reproduces supported spatial transformations. MRI is not interpreted as HU data; select an MRI sequence view instead. Stored image evidence includes the source metadata and the observation ID that used it.

Reference reports, positive outcome labels and disease lists remain in loader samples for evaluation. `get_exam_info()` excludes these benchmark targets. It supplies available clinical/acquisition context and, after sample loading, the actual `preview_metadata` for the input image.

## Observation and evidence records

`link_visual_evidence` creates a stable observation ID alongside the image binding. Observations retain anatomy, laterality, polarity, attributes, certainty, confidence and image IDs. Pass that ID to `standardize_observation` to add RadLex/SNOMED normalization without creating a duplicate observation or discarding negation and uncertainty. Explicit structured attributes take precedence over conservative text-derived defaults; the original text is always retained.

Example tool arguments, using identifiers returned by the preceding tools:

```json
{"observation_text": "No left pleural effusion", "observation_id": "OBS_returned_by_visual_tool"}
```

`concept_lookup`, `literature_search` and `case_retrieval` accept `observation_ids`. Use these to record which observations motivated a query. Unknown IDs fail validation. Retrieved documents receive stable IDs derived from source identity, content and snapshot version; repeat retrieval reuses the record and merges explicit observation associations. Terminology lookup now produces bindable retrieved evidence as well as displaying search results.

Retrieval relevance is not a support verdict. New evidence is marked `unassessed`; claim-specific clinical/evidence review decides whether the evidence supports the generated claim. Similar historical cases are explicitly marked as not establishing patient-specific findings. Summaries retain observation-specific associations, original wording, polarity, laterality and unresolved conflicts.

When a visual or retrieval record changes during remediation, its descendants enter the affected set and their previous reviews are invalidated. Existing observations outside that set cannot be rewritten. Updating an observation replaces its current image dependencies; unaffected observations sharing an image retain their separate records.

## Frozen retrieval corpus

Evolution must search a materialized corpus rather than live PubMed or a remote guideline/case service. `CorpusSnapshot` is a queryable document collection, not a cache of fixed query results, so the two replay arms can issue different queries against identical underlying documents.

Prepare records from verified, licensed source documents. Every record requires `document_id`, `source_type` and `content`. Optional fields include `title`, `category` and source-specific `metadata`. Allowed source types are `radlex`, `snomed`, `guideline`, `pubmed`, `literature` and `case`. Case records also require a canonical source `patient_id` and `partition: "train"`. Designated final-evaluation patients must be passed as `excluded_patient_ids`.

A materialization call using previously prepared records is:

```python
from src.kb.snapshot import CorpusSnapshot

snapshot = CorpusSnapshot.materialize(
    "data/snapshots/corpus.json",
    records=verified_document_records,
    sources=["radlex", "snomed", "guideline", "pubmed", "case"],
    excluded_patient_ids=final_evaluation_source_patient_ids,
    provenance=source_versions_and_licences,
)
print(snapshot.version)
```

This refuses to overwrite an existing snapshot. Declared sources may have zero records, but undeclared sources cannot be queried. The reporting registry requires both terminology sources, cases, and at least one guideline/literature source to be declared. Partition and patient-exclusion checks reject ineligible case documents. Content changes invalidate the SHA-256 version.

Configure a frozen run with the returned version:

```yaml
retrieval:
  snapshot_path: ./data/snapshots/corpus.json
  snapshot_version: THE_RETURNED_SHA256
  require_snapshot: true
```

The registry uses the configured local embedding model in frozen mode and does not download replacement weights or fall back to live services. Search combines normalized dense cosine ranking with BM25 through reciprocal-rank fusion with constant 60. Direct `CorpusSnapshot` use without an embedder provides explicitly identified lexical BM25 retrieval for offline checks; this should not be reported as a dense-retrieval experiment. The existing final evidence reranker retains cosine/lexical weights 0.6/0.4 and the 512-character evidence embedding input, with lexical fallback if reranking embedding inference fails.

Actual snapshot documents, source provenance, canonical patient allocations and local embedding weights must be supplied by the experiment. No corpus, clinical outcomes or patient assignments are synthesized by these code changes.

## Verification

`tests/test_method_evidence_images.py` exercises the shared view budget, source metadata and transport dimensions, physical CT windowing, real NIfTI preview axes, coarse-to-target access, sequence/range rejection, negation and laterality preservation, dependency invalidation, frozen-corpus hash integrity, training-only case records and network-free snapshot retrieval. The existing image, evidence and volumetric-loader tests remain applicable. These are local implementation tests; they do not establish clinical accuracy.
