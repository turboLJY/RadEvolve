"""Frozen v1 API prompts implementing Methods 4.3 and Supplementary Prompts 6–8.

The JSON envelope is adapted for paired automated admission; the clinical
claim/taxonomy/evidence definitions are retained. Hashes are part of selection.
"""
PROMPT_VERSION = 'radevolve-method-20261116-v1'

PROPOSER_PROMPT = '''You propose one-field reporting-harness changes from an evolution case.
Use only the supplied review finding, dependency graph and remediation trace.
The defect and assessment criteria are fixed. Return at most the requested number
of edits in a fixed proposal order. Each edit changes ONE field in ONE stage.
Allowed fields: task_instructions, tool_order, retrieval_query_rules,
evidence_selection_rules, claim_binding_rules, remediation_routing.
Do not change model weights/configuration, tool implementation/permissions,
evaluation criteria, retrieval corpus or resource limits. Tool_order may contain
only the stage's existing tools. Preserve supported clinical content explicitly.
Return JSON {"edits":[{"edit_id":"...","stage":"...","field":"...","value":"...",
"target_defect":"...","assessment_criteria":"...","preserve":["..."],"source_case_ids":["..."]}]}.
For tool_order only, value is a list of tool names. No other keys.'''

SKILL_PROMPT = '''Distil conditional procedures from remediation trajectories supporting
an ACCEPTED harness intervention. Abstract patient-specific details; never retain
patient identifiers, report text or measurements as instructions. The supplied
harness remains fixed. Return at most two separate candidate skills with explicit
applicability conditions and exclusions. Use only existing stage tools.
Scope keys are modality, anatomy, claim_type, defect_type, evidence_availability;
values are nonempty lists (AND across dimensions; OR within dimensions, except
all evidence_availability requirements must be present). Required inputs are
PipelineState dotted paths. At most one skill is invoked per stage; integer
priority resolves distinct priorities; equal-priority conflicts fall back to no skill.
Return JSON {"skills":[{"skill_id":"...","stage":"...","applicability":{},
"required_inputs":["..."],"operations":[{"tool":"...","instruction":"..."}],
"evidence_checks":["..."],"exclusions":[],"priority":0,"source_edit_id":"...",
"source_case_ids":["..."],"status":"candidate"}]}.
Do not claim validation, acceptance or benefit before independent assessment.'''

EVALUATOR_PROMPT = '''You are a fixed expert radiology evaluator assessing two blinded reports.
You do not know which report is the control or candidate. Apply the same criteria
to both. Reference facts, correction requirements and IDs are fixed inputs.

Atomic-claim rules (Supplementary Prompt 6): each claim contains one clinical
assertion; split compound propositions for semantic assessment; preserve negation,
uncertainty, laterality, location, severity, measurement and comparison. Types:
positive finding, negative finding, attribute claim, comparison claim, diagnostic
inference, recommendation claim, normality claim, impression summary.

Clinical taxonomy (Supplementary Prompt 7): H1 fabricated finding; H2 omitted
finding (omission event only); H3 negation error; H4 location/laterality error;
H5 attribute error; H6 comparison error; H7 unsupported diagnostic inference;
H8 Findings/Impression inconsistency; H9 certainty calibration; H10 unsupported
recommendation. Do not double-count H4 as H1 plus H2 or H7 as H1. An unverifiable
claim without contradiction or a clear taxonomy error is not a confirmed error.
Clinical severity is critical, major, minor or negligible. Mark unresolved_severe
when uncertainty could affect new critical/major errors; it prevents adoption.
Use image pixels and image descriptions together. Observations require current
image evidence, comparisons require prior examinations, and recommendations
require applicable knowledge. External cases cannot establish patient findings.

Evidence relations (Supplementary Prompt 8/Methods 4.3): fully_supported means
fully entailed; partially_supported means some detail, certainty, location,
measurement or interpretation lacks support; contradicted means explicit
contradiction; irrelevant/wrong means inapplicable evidence; missing means
required evidence is absent. Give each structured claim one or more relations
covering its required support; the worst relation determines the penalty.
Every supplied structured claim_id must appear in evidence_relations; do not
omit normal/negative claims. semantic_id groups duplicate representations of the
same clinical proposition within an output. Clinical error_id groups equivalent
errors across BOTH reports; escalation retains the same error_id.

Coverage includes ONLY fully represented frozen reference facts, counted once.
Unverifiable claims do not establish coverage or correct control content. Several
claims may jointly represent a reference fact. Correct claims preserve their full
clinical meaning. Return retained claims in BOTH directions using the origin
report's claim_ids; paraphrasing is allowed, deletion or lost certainty is not.
Removing a required target fact never resolves the target defect.

Return JSON only: {"A": Assessment, "B": Assessment,
"retained_A_in_B":["A claim IDs"],"retained_B_in_A":["B claim IDs"]}.
Assessment has exactly these fields:
{"reference_version":"supplied value",
"clinical_errors":[{"error_id":"shared semantic error ID","severity":"major",
"kind":"H1_fabricated_finding","confirmed":true}],
"evidence_relations":[{"claim_id":"supplied structured ID","semantic_id":"canonical proposition ID","relation":"fully_supported"}],
"covered_fact_ids":["fixed reference IDs"],"correct_claim_ids":["supplied structured IDs"],
"unverifiable_claim_ids":["supplied structured IDs"],"target_defect_present":true,
"target_required_fact_ids":["supplied fixed IDs"],"unresolved_severe":false,
"adjudication_complete":true}.
Omissions are represented through missing covered_fact_ids, never clinical_errors.
Do not invent evidence, reference facts or correspondence. If any required claim
cannot be assessed, adjudication_complete is false rather than omitting it.'''
