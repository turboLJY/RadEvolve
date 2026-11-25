"""Shared evidence identifiers and explicit observation-to-retrieval dependencies."""
from __future__ import annotations

import hashlib
import json


def stable_id(prefix: str, payload) -> str:
    return prefix + "_" + hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def observation_ids(args: dict, state) -> list[str]:
    ids = list(dict.fromkeys(args.get("observation_ids", [])))
    known = {o.get("observation_id") for o in state.observation_graph} if state else set()
    if any(i not in known for i in ids):
        raise ValueError("observation_ids contains an unknown observation")
    return ids


def prepare_record_update(state, identifier: str, kind: str, payload: dict):
    """Invalidate dependent claims before replacing a case-local record."""
    graph = getattr(state, "dependency_graph", None)
    if graph is None:
        return
    previous = graph.nodes.get(identifier)
    changed = previous is None or previous.get("payload") != payload
    if not changed:
        return
    if state.iteration_count > 1:
        if previous is not None and identifier not in state.affected_node_ids:
            raise ValueError(f"Record {identifier} is outside the dependency repair set")
        affected = graph.invalidate({identifier}) if previous is not None else {identifier}
        state.affected_node_ids.update(affected)
        state.reviewed_claim_ids.difference_update(affected)
        state.reviewed_observation_ids.difference_update(affected)
        for node_id in affected:
            state.claim_reviews.pop(node_id, None)
        state.repaired_node_ids.add(identifier)
    state.review_completed = False
    state.review_passed = False
    state.natural_language_report = ""
    state.rendered_claim_spans = []


def add_evidence(state, evidence, inputs: list[str]):
    if state is None:
        return
    evidence.metadata["observation_ids"] = sorted(set(inputs) | set(evidence.metadata.get("observation_ids", [])))
    existing = next((e for e in state.retrieved_evidence if e.evidence_id == evidence.evidence_id), None)
    if existing:
        evidence.metadata["observation_ids"] = sorted(set(evidence.metadata["observation_ids"]) | set(existing.metadata.get("observation_ids", [])))
        prepare_record_update(state, evidence.evidence_id, "evidence", evidence.to_dict())
        state.retrieved_evidence[state.retrieved_evidence.index(existing)] = evidence
    else:
        prepare_record_update(state, evidence.evidence_id, "evidence", evidence.to_dict())
        state.retrieved_evidence.append(evidence)
    graph = getattr(state, "dependency_graph", None)
    if graph:
        graph.add_node(evidence.evidence_id, "evidence", evidence.to_dict())
        for obs_id in evidence.metadata["observation_ids"]:
            graph.add_edge(obs_id, evidence.evidence_id)
