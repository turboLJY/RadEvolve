These are **synthetic format templates**, not patients, evidence, costs or measured outcomes.
The empty corpus and null baseline costs cannot establish clinical benefit or admission.

Run a no-API software smoke test from the repository root:

```bash
python scripts/run_evolution.py demo --output-dir output/evolution_synthetic
```

For real offline evolution, create a patient-disjoint manifest with fixed reference
facts, actual image/sequence paths and reporting-time clinical context. The defect
file fixes the target definition and required reference-fact IDs separately for
each evolution/replay case and every development-validation case before replay.
An empty required-fact list is appropriate only when that case has no applicable
target-required fact; it must still be explicitly supplied. Final cases must be
excluded from source patient IDs in the materialized retrieval corpus. Never put
reference reports, labels, outcomes or diagnoses from benchmark metadata in the
clinical context. Set `retrieval.snapshot_path`, `require_snapshot: true`, and
archived CNY input/output tariffs in the settings file. Do not guess prices.

The following commands use the configured model API and incur inference costs:

```bash
python scripts/run_evolution.py measure-baseline --settings config/settings.yaml \
  --manifest /path/to/actual_manifest.json --output output/h0_costs.json
python scripts/run_evolution.py run --settings config/settings.yaml \
  --manifest /path/to/actual_manifest.json --defects /path/to/fixed_defects.json \
  --baseline-costs output/h0_costs.json --output-dir output/evolution
```

Baseline measurement deliberately uses the same three complete executions per
case and development cohort as acceptance. It averages runs within source
patients, then gives patients equal weight; that measured reference cost is fixed
throughout selection. This software protocol choice makes the cost comparison
reproducible and does not imply that any real baseline has already been measured.

Selection writes accepted/rejected records, pairs, checkpoints, hashes and model
usage to its output directory. `harness.json`, `skills.json` and
`frozen_manifest.json` can be supplied via the `evolution` runtime settings for
final reporting. Skills are never learned from the final reporting cases.

```bash
python scripts/run_evolution.py verify-freeze output/evolution/frozen_manifest.json
python scripts/run_evolution.py audit \
  --records output/evolution_synthetic/records/0001_harness_acceptance.json \
  --reference-cost-cny 1 --case-ids synthetic_case_1,synthetic_case_2 \
  --output output/evolution_synthetic/recomputed_decision.json
```
