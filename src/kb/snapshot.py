"""Immutable, content-addressed retrieval corpus for paired evolution runs.

A snapshot materializes documents (not query/response caches): both replay arms
can issue different queries against the same corpus. No method in this module
performs network access. The manifest declares the sources, including empty ones.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from src.kb.base import KBSearchResult, KnowledgeSource


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


class CorpusSnapshot:
    SCHEMA_VERSION = 1
    ALLOWED_SOURCES = {"radlex", "snomed", "guideline", "pubmed", "literature", "case"}

    def __init__(self, path: str | Path, expected_version: str = "", embedder=None):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("Unsupported retrieval snapshot schema_version")
        self.sources = set(payload.get("sources", []))
        if not self.sources or not self.sources <= self.ALLOWED_SOURCES:
            raise ValueError("Snapshot must explicitly declare valid sources")
        self.records = payload.get("records")
        if not isinstance(self.records, list):
            raise ValueError("Snapshot records must be a list")
        ids = set()
        excluded = set(payload.get("excluded_patient_ids", []))
        for record in self.records:
            if not isinstance(record, dict) or not record.get("document_id") or not isinstance(record.get("content"), str):
                raise ValueError("Each snapshot document requires document_id and content")
            key = record["document_id"]
            if key in ids:
                raise ValueError(f"Duplicate snapshot document_id: {key}")
            ids.add(key)
            if record.get("source_type") not in self.sources:
                raise ValueError("Document source must be declared in snapshot sources")
            if record.get("source_type") == "case":
                if not record.get("patient_id"):
                    raise ValueError("Case documents require patient_id for leakage checks")
                if record["patient_id"] in excluded or record.get("partition") != "train":
                    raise ValueError("Evaluation patients/partitions are prohibited in the retrieval corpus")
        self.version = hashlib.sha256(_canonical({k: v for k, v in payload.items() if k != "version"})).hexdigest()
        if payload.get("version") and payload["version"] != self.version:
            raise ValueError("Retrieval snapshot content hash mismatch")
        if expected_version and expected_version != self.version:
            raise ValueError("Retrieval snapshot differs from frozen configuration")
        self._embedder = embedder

    @classmethod
    def materialize(cls, path: str | Path, records: list[dict], sources: list[str], *, excluded_patient_ids=(), provenance: dict | None = None):
        payload = {"schema_version": cls.SCHEMA_VERSION, "sources": sorted(set(sources)),
                   "records": records, "excluded_patient_ids": sorted(set(excluded_patient_ids)),
                   "provenance": provenance or {}}
        payload["version"] = hashlib.sha256(_canonical(payload)).hexdigest()
        path = Path(path)
        if path.exists():
            raise FileExistsError("Refusing to replace a frozen retrieval snapshot")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        try:
            return cls(path)
        except Exception:
            path.unlink()
            raise

    @staticmethod
    def _tokens(text: str):
        return re.findall(r"[a-z0-9]+", text.lower())

    def search(self, query: str, source_types, category: str = "", max_results: int = 5) -> list[dict]:
        source_types = set(source_types)
        missing = source_types - self.sources
        if missing:
            raise ValueError(f"Frozen corpus has no declared source(s): {sorted(missing)}")
        records = [r for r in self.records if r["source_type"] in source_types
                   and (not category or r.get("category", category) in {category, "all"})]
        if not records:
            return []
        tokens = [Counter(self._tokens(r["content"] + " " + r.get("title", ""))) for r in records]
        lengths = [sum(t.values()) for t in tokens]
        avg_len = sum(lengths) / len(lengths) or 1.0
        query_tokens = set(self._tokens(query))
        lexical = []
        for i, counts in enumerate(tokens):
            score = 0.0
            for token in query_tokens:
                freq = counts[token]
                df = sum(1 for t in tokens if token in t)
                idf = math.log(1 + (len(tokens) - df + 0.5) / (df + 0.5))
                denominator = freq + 1.5 * (1 - 0.75 + 0.75 * lengths[i] / avg_len)
                score += idf * freq * 2.5 / denominator if denominator else 0
            lexical.append(score)
        lexical_rank = sorted(range(len(records)), key=lambda i: (-lexical[i], records[i]["document_id"]))
        fused = {i: 1 / (60 + rank) for rank, i in enumerate(lexical_rank, 1)}
        if self._embedder is not None:
            # Embeddings use the same fixed local model for both arms.
            q = np.asarray(self._embedder.embed_query(query), dtype=float)
            vectors = np.asarray(self._embedder.embed_texts([r["content"] for r in records]), dtype=float)
            norms = np.linalg.norm(vectors, axis=1)
            scores = (vectors @ q) / np.maximum(norms * np.linalg.norm(q), 1e-12)
            dense_rank = sorted(range(len(records)), key=lambda i: (-scores[i], records[i]["document_id"]))
            for rank, i in enumerate(dense_rank, 1):
                fused[i] += 1 / (60 + rank)
        else:
            # With no embedding model, report lexical retrieval explicitly.
            fused = {i: lexical[i] for i in range(len(records)) if lexical[i] > 0}
        order = sorted(fused, key=lambda i: (-fused[i], records[i]["document_id"]))[:max_results]
        return [{**records[i], "score": fused[i], "snapshot_version": self.version,
                 "retrieval_mode": "snapshot_rrf" if self._embedder is not None else "snapshot_bm25"} for i in order]


class SnapshotKnowledgeSource(KnowledgeSource):
    def __init__(self, snapshot: CorpusSnapshot, source: str):
        if source not in snapshot.sources:
            raise ValueError(f"Frozen corpus does not declare {source}")
        self.snapshot, self.source = snapshot, source

    def _result(self, r):
        return KBSearchResult(r["document_id"], self.source, r.get("title", ""), r["content"],
                              r.get("score", 0), {**r.get("metadata", {}), "snapshot_version": self.snapshot.version})

    async def search(self, query: str, max_results: int = 10):
        return [self._result(r) for r in self.snapshot.search(query, [self.source], max_results=max_results)]

    async def lookup(self, code: str):
        r = next((r for r in self.snapshot.records if r["document_id"] == code and r["source_type"] == self.source), None)
        return self._result(r) if r else None

    def is_available(self):
        return True
