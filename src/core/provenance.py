"""Case-local dependencies and durable, schema-validated execution records.

Identifiers are scoped to a case. SQLite stores immutable revisions, so updating
an observation does not erase the evidence available at an earlier checkpoint.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Iterable


RECORD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["schema_version", "session_id", "record_id", "kind", "payload"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": "1"},
        "session_id": {"type": "string", "minLength": 1},
        "record_id": {"type": "string", "minLength": 1},
        "kind": {"enum": ["image", "observation", "evidence", "finding", "impression",
                           "tool_call", "review", "checkpoint", "stage", "result", "request"]},
        "payload": {"type": "object"},
    },
}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class DependencyGraph:
    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: set[tuple[str, str]] = set()
        self.invalidated: set[str] = set()

    def add_node(self, node_id: str, kind: str, payload: dict | None = None) -> None:
        if not isinstance(node_id, str) or not node_id:
            raise ValueError("A dependency node needs a non-empty stable identifier")
        self.nodes[node_id] = {"kind": kind, "payload": copy.deepcopy(payload or {})}
        self.invalidated.discard(node_id)

    def add_edge(self, source: str, target: str) -> None:
        if source not in self.nodes or target not in self.nodes:
            raise ValueError(f"Unknown dependency endpoint: {source} -> {target}")
        if source == target or source in self.descendants({target}):
            raise ValueError(f"Dependency cycle: {source} -> {target}")
        self.edges.add((source, target))

    def set_dependencies(self, target_id: str, source_ids: Iterable[str]) -> None:
        previous = set(self.edges)
        self.edges = {(a, b) for a, b in self.edges if b != target_id}
        try:
            for source in set(source_ids):
                self.add_edge(source, target_id)
        except Exception:
            self.edges = previous
            raise

    def descendants(self, node_ids: Iterable[str]) -> set[str]:
        roots = set(node_ids)
        found: set[str] = set()
        pending = list(roots)
        while pending:
            source = pending.pop()
            for a, b in self.edges:
                if a == source and b not in found and b not in roots:
                    found.add(b)
                    pending.append(b)
        return found

    def ancestors(self, node_ids: Iterable[str]) -> set[str]:
        roots = set(node_ids)
        found: set[str] = set()
        pending = list(roots)
        while pending:
            target = pending.pop()
            for a, b in self.edges:
                if b == target and a not in found and a not in roots:
                    found.add(a)
                    pending.append(a)
        return found

    def invalidate(self, node_ids: Iterable[str]) -> set[str]:
        affected = set(node_ids)
        affected |= self.descendants(affected)
        self.invalidated |= affected
        return affected

    def clear_invalidated(self, node_ids: Iterable[str]) -> None:
        self.invalidated.difference_update(node_ids)

    def to_dict(self) -> dict:
        return {"nodes": copy.deepcopy(self.nodes), "edges": [list(e) for e in sorted(self.edges)],
                "invalidated": sorted(self.invalidated)}

    @classmethod
    def from_dict(cls, value: dict) -> "DependencyGraph":
        graph = cls()
        for node_id, node in value.get("nodes", {}).items():
            graph.add_node(node_id, node["kind"], node.get("payload", {}))
        for source, target in value.get("edges", []):
            graph.add_edge(source, target)
        graph.invalidated = set(value.get("invalidated", []))
        if not graph.invalidated <= graph.nodes.keys():
            raise ValueError("Unknown invalidated node")
        return graph


class SQLiteRecordStore:
    """Append-only JSON revisions; content hashes identify frozen records."""

    def __init__(self, path: str | Path):
        from jsonschema import Draft202012Validator

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.validator = Draft202012Validator(RECORD_SCHEMA)
        self.connection = sqlite3.connect(str(path))
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS records (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, record_id TEXT NOT NULL, kind TEXT NOT NULL, "
            "content_hash TEXT NOT NULL, document TEXT NOT NULL, "
            "created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')))"
        )
        self.connection.execute("CREATE INDEX IF NOT EXISTS records_case ON records(session_id, record_id)")
        self.connection.commit()

    def append(self, session_id: str, record_id: str, kind: str, payload: dict) -> str:
        document = {"schema_version": "1", "session_id": session_id, "record_id": record_id,
                    "kind": kind, "payload": payload}
        self.validator.validate(document)
        serialized = canonical_json(document)
        digest = hashlib.sha256(serialized.encode()).hexdigest()
        with self.connection:
            self.connection.execute(
                "INSERT INTO records(session_id, record_id, kind, content_hash, document) VALUES (?, ?, ?, ?, ?)",
                (session_id, record_id, kind, digest, serialized),
            )
        return digest

    def read(self, session_id: str, record_id: str | None = None) -> list[dict]:
        sql = "SELECT document FROM records WHERE session_id=?"
        values = [session_id]
        if record_id is not None:
            sql += " AND record_id=?"
            values.append(record_id)
        return [json.loads(row[0]) for row in self.connection.execute(sql + " ORDER BY sequence", values)]

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
