# RadEvolve Method implementation

This implementation follows the supplied `RadEvolve.pdf`, particularly Methods 4.1–4.3 and 4.10. It modifies the runnable project, not the manuscript. The model weights remain unchanged. Software tests and synthetic examples are not clinical performance results.

## Method-to-code mapping

| Method requirement | Implementation |
| --- | --- |
| Atomic claims with anatomy, laterality, polarity, attributes, certainty and evidence | `src/core/types.py`, `src/report/schema.py`, `src/tools/report/` |
| Observation / comparison / recommendation evidence requirements | `src/report/grounding.py`; current image support, actual supplied priors, applicable knowledge respectively |
| Exact structured-to-text mapping | `render_claims`; `rendered_claim_spans` stores claim ID, section and end-exclusive character ranges |
| Case-local directed dependency graph | `src/core/provenance.py`; explicit consumed/produced IDs, replaced dependencies and transitive invalidation |
| Stable tool, observation, evidence and claim records | `src/core/execution_records.py`; append-only SQLite JSON revisions, JSON Schema validation and SHA-256 hashes |
| Attribution-guided claim remediation | `src/core/orchestrator.py`; retain issue feedback, rerun responsible/downstream stages, restrict edits to affected claims |
| Complete review before PASS | `record_claim_review`, omission coverage and deterministic controller integrity checks; unresolved reports retain REVISE after at most three passes |
| Real visual reinspection | `src/core/loop.py` sends tool-generated PNGs back to the VLM; review receives current claim-linked views |
| 32 views across sequences and passes | `src/data/image_store.py`, `src/tools/visual/select_slice.py`; source localization and transformations remain available |
| Fixed retrieval corpus during evolution | `src/kb/snapshot.py`; versioned documents, patient/partition exclusions, local queries and no live fallback |
| One-field harness proposals | `src/evolution/schemas.py`, `models.py`, `prompts.py`; fixed permissions and allowed instruction/policy fields |
| Independent paired checkpoint replay | `src/evolution/runner.py`; explicit JSON codec, case-input hashes and configuration-version checks |
| Fixed-reference loss and admission statistics | `src/evolution/evaluation.py`; clinical, evidence and omission components plus patient-level paired bootstrap |
| Offline update budget and adoption | `src/evolution/engine.py`; local screening, independent complete validation, version history and rollback |
| Conditional skill library | `src/evolution/runtime.py`, `schemas.py`, `engine.py`; deterministic applicability, one skill per stage, conflict fallback and regression deactivation |
| Development/final separation and freezing | `src/evolution/manifest.py`; canonical patient allocations, immutable component hashes and final-configuration verification |

Local repair retains the same stage architecture. A stage may run again while only affected structured records can be replaced. New dependencies can expand the affected set. Re-review does not count as recomputing an invalidated claim, and merely rebinding stale evidence cannot make it valid. Current evidence records remain distinct from model judgements about whether that evidence clinically supports a claim.

## Prespecified values

Execution defaults are Qwen3.6-Flash, temperature 0.7, top-p 0.95 and 8,192 output tokens for all four agents. Their model-turn limits are 50/20/20/15. A turn limit allows one final tool-free completion. Transient API failures receive at most five retries beyond the initial request; SDK retries are disabled to avoid multiplying that budget. Content errors are not retried as transient failures. Missing/failed required runs cannot pass admission.

Harness search permits five rounds, at most four candidate edits per selected defect and at most 20 entries into local screening, including failures and reassessments. Every candidate uses five local paired repetitions; development acceptance uses three complete paired executions per case. A skill generator may propose at most two candidate skills per accepted intervention.

Loss uses a fixed number of reference facts per case and weights `(0.50, 0.25, 0.25)`. Non-omission clinical errors have severity weights 1/0.5/0.1/0 for critical/major/minor/negligible. The worst required evidence relation contributes 0/0.5/1 for full/partial/failed support. Clinical and evidence losses are not clipped at one. Omissions enter only the omission component. Duplicate semantic errors/propositions are counted once in each component. Unverifiable claims cannot establish correctness or reference coverage.

Admission averages repeated runs within each patient before equally weighting patients. The defaults are 20,000 patient-cluster bootstrap resamples, seed 20260615 and a one-sided 95% lower confidence bound. A missing retention value is excluded only from retention; bootstrap draws with no estimable retention are excluded from its bound and their count is reported. A cohort with no estimable retention cannot pass. Lower bounds must satisfy:

- loss reduction ≥ 0.01;
- correct-claim retention ≥ 0.99;
- reference-coverage difference ≥ −0.01.

Any newly introduced major/critical error, unresolved potentially severe error, incomplete required run or unestimable metric prevents adoption. The candidate mean complete-run cost must not exceed 1.25 times the measured initial `h0,S0` cost. This initial reference cost remains fixed across rounds. Harness admission also requires fewer occurrences of the prespecified defect. Removing its required clinical fact does not count as correction.

## Local software checks

Install the project and its development extra in a Python 3.10+ environment:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests -q
python scripts/run_evolution.py demo --output-dir output/evolution_synthetic
```

The `demo` command uses explicitly synthetic runner/evaluator fixtures. It exercises screening, complete validation, bootstrap gates and version persistence without model calls or patient data. The test suite separately exercises the actual reporting loop and tools using a scripted VLM, plus checkpoint codecs, evidence/source constraints and rejection cases.

Recompute a saved acceptance decision without an API:

```bash
python scripts/run_evolution.py audit \
  --records output/evolution_synthetic/records/0001_harness_acceptance.json \
  --reference-cost-cny 1 \
  --case-ids synthetic_case_1,synthetic_case_2 \
  --output output/evolution_synthetic/audited_decision.json
```

The cost of 1 in this example belongs only to the synthetic fixture.

## Real offline evolution

Supply experiment-specific inputs before invoking `run`:

1. A patient manifest with explicit evolution, development and final allocations, canonical source-patient identifiers and reporting-time clinical context. Evolution and development cases must be separate training patients. Shared patients across derivative datasets must use the same canonical identity. Cases with zero reference facts are listed as unevaluable before generation.
2. Deduplicated, fixed reference facts and prespecified target-defect assessment criteria, including required target-fact IDs for every replay and development case. Reference facts are evaluator inputs and are excluded from reporting prompts.
3. A materialized, versioned retrieval corpus and the local embedding assets. [Image and retrieval instructions](image_retrieval_alignment.md) describe the document schema and patient exclusions. Queries may differ between arms; the corpus may not change.
4. Measured initial `h0,S0` complete-run costs and the archived input/output tariff used by the experiment. Prices are intentionally `null` in the checked-in configuration. Request records retain usage and unknown costs; unknown cost does not satisfy the cost gate.
5. The configured reporting/proposal/skill/evaluation API endpoints and permissions. The provided implementation uses Qwen3.6-Max-Preview for acceptance adjudication; the evaluator receives blinded report labels and applies fixed versioned prompts.

```bash
python scripts/run_evolution.py run \
  --settings config/settings.yaml \
  --manifest path/to/patient_manifest.json \
  --defects path/to/defects.json \
  --baseline-costs path/to/measured_h0_costs.json \
  --output-dir output/evolution
```

The [synthetic schema templates](../src/evolution/examples/README.md) show the required input formats. Replace them with verified experiment data; they are not ready-to-run patient records. To measure `B_ref` on the same complete development cohort before selection, use:

```bash
python scripts/run_evolution.py measure-baseline \
  --settings config/settings.yaml \
  --manifest path/to/patient_manifest.json \
  --output path/to/measured_h0_costs.json
```

This measurement command also invokes the real reporting API and requires configured archived prices. Its output records all required case/repetition costs, canonical patient IDs, the manifest hash and initial model/corpus/resource/harness/library versions. Selection refuses incomplete or mismatched baseline records.

This command executes real model calls. The current code change does not run this experiment, create clinical references, assign actual patients, infer service tariffs or claim any accepted clinical improvement. The evaluator remains a model-based assessment whose correctness must be established separately.

Outputs include `selection_protocol.json`, current `harness.json`/`skills.json`, immutable version records, checkpoints, accepted/rejected candidate decisions, paired reports and scores, request accounting and `frozen_manifest.json`. Failed-run costs and proposal/skill/evaluation workloads are recorded separately from complete reporting costs. Components referenced by the frozen manifest must remain available.

## Use selected versions for reporting

Add these paths to the existing YAML settings after a completed selection run:

```yaml
evolution:
  harness_path: ./output/evolution/harness.json
  skill_library_path: ./output/evolution/skills.json
  frozen_manifest_path: ./output/evolution/frozen_manifest.json
retrieval:
  snapshot_path: ./data/snapshots/corpus.json
  snapshot_version: THE_VERIFIED_SNAPSHOT_SHA256
  require_snapshot: true
```

Verify the saved artifact bundle:

```bash
python scripts/run_evolution.py verify-freeze output/evolution/frozen_manifest.json
```

Normal `run_single.py`, `run_trace.py` and full-pipeline runners load the selected harness/library once per case. Case-specific memory is fresh for every original input. Within-case remediation remains active, and it does not modify persistent procedures. Empty harness/library paths select the initial workflow. Do not interpret the old `direct`, `visual_only` or `retrieval_only` component ablations as the paper's A/B/C harness/skill configurations: A uses initial harness/no skills; B uses final harness/no skills; C uses final harness/final skills. Each final configuration needs its own matching frozen record.

The report's `review_status` is `PASS` or `REVISE`; `completed` records whether required execution completed, not clinical correctness. Baselines without review are `NOT_REVIEWED`. Consumers should retain unresolved issues and inspect `rendered_claim_spans`, `review_history`, `dependency_graph`, `execution_trace`, `versions` and the SQLite `record_database` path.
