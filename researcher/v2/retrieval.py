
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
SCHEMA_VERSION = 2
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RECORD_COLUMNS = (
    "record_id",
    "document_id",
    "record_type",
    "node_id",
    "parent_id",
    "node_title",
    "section_path",
    "content_block_id",
    "searchable_text",
    "source_text",
    "block_type",
    "start_line",
    "end_line",
    "embedding",
)

DOCUMENT_COLUMNS = (
    "document_id",
    "doc_name",
    "doc_title",
    "structure_path",
    "record_count",
    "indexed_at",
)

METADATA_COLUMNS = ("key", "value")


# ---------------------------------------------------------------------
# Structure traversal and record preparation
# ---------------------------------------------------------------------


def walk_tree(
    nodes: list[dict[str, Any]],
    parent_id: str | None = None,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[dict[str, Any], str | None, tuple[str, ...]]]:
    """Yield each node with its parent ID and human-readable section path."""

    for node in nodes:
        if not isinstance(node, dict):
            raise ValueError("Every structure node must be an object.")

        children = node.get("children")
        if children is None:
            children = []

        if not isinstance(children, list):
            node_id = node.get("node_id") or "<unknown>"
            raise ValueError(f"Node {node_id!r}: children must be a list.")

        title = str(node.get("title") or "Untitled section")
        current_path = path + (title,)

        yield node, parent_id, current_path

        yield from walk_tree(
            children,
            node.get("node_id"),
            current_path,
        )


def tokenize(text: str) -> list[str]:
    """Simple deterministic tokenizer that preserves technical tokens."""
    return re.findall(
        r"[a-zA-Z0-9]+(?:[._/-][a-zA-Z0-9]+)*",
        text.lower(),
    )


def _record_id(
    document_id: str,
    record_type: str,
    node_id: str,
    block_id: str | None = None,
) -> str:
    """Generate a deterministic, document-scoped record identity."""

    identity = json.dumps(
        [document_id, record_type, node_id, block_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )

    return f"record:{identity}"


def make_records(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Create document-owned navigation and original-evidence records."""

    if not isinstance(tree, dict):
        raise ValueError("structure.json must contain a JSON object.")

    document_id = tree.get("document_id")

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError(
            "structure.json must contain a stable document_id."
        )

    structure = tree.get("structure")

    preamble = tree.get("preamble")
    if preamble is None:
        preamble = []

    if not isinstance(structure, list):
        raise ValueError("'structure' must be a list.")

    if not isinstance(preamble, list):
        raise ValueError("'preamble' must be a list.")

    doc_title = str(
        tree.get("doc_title")
        or tree.get("doc_name")
        or "Untitled document"
    )

    records: list[dict[str, Any]] = []

    seen_nodes: set[str] = set()
    seen_record_ids: set[str] = set()

    def append_record(record: dict[str, Any]) -> None:
        record_id = record["record_id"]

        if record_id in seen_record_ids:
            raise ValueError(
                f"Duplicate retrieval record identity: {record_id}"
            )

        seen_record_ids.add(record_id)
        records.append(record)

    # -------------------------------------------------------------
    # Navigation and evidence records from the section tree
    # -------------------------------------------------------------

    for node, parent_id, section_path in walk_tree(structure):

        node_id = node.get("node_id")

        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError(
                "Every structure node must have a non-empty node_id."
            )

        if node_id == "__preamble__":
            raise ValueError(
                "'__preamble__' is reserved for preamble evidence."
            )

        if node_id in seen_nodes:
            raise ValueError(f"Duplicate node_id: {node_id}")

        seen_nodes.add(node_id)

        content_blocks = node.get("content_blocks")
        if content_blocks is None:
            content_blocks = []

        children = node.get("children")
        if children is None:
            children = []

        key_items = node.get("key_items")
        if key_items is None:
            key_items = []

        if not isinstance(content_blocks, list):
            raise ValueError(
                f"Node {node_id!r}: content_blocks must be a list."
            )

        if not isinstance(children, list):
            raise ValueError(
                f"Node {node_id!r}: children must be a list."
            )

        if not isinstance(key_items, list):
            raise ValueError(
                f"Node {node_id!r}: key_items must be a list."
            )

        title = str(node.get("title") or "Untitled section")
        summary = str(node.get("summary") or "")

        key_text = "; ".join(
            str(item) for item in key_items
        )

        nav_text = "\n".join(
            part
            for part in (
                f"Document: {doc_title}",
                f"Section: {title}",
                f"Summary: {summary}",
                f"Key concepts: {key_text}",
            )
            if part.strip()
        )

        if summary or key_text:
            append_record({
                "record_id": _record_id(
                    document_id,
                    "navigation",
                    node_id,
                ),
                "document_id": document_id,
                "record_type": "navigation",
                "node_id": node_id,
                "parent_id": parent_id,
                "node_title": title,
                "section_path": list(section_path),
                "content_block_id": None,
                "text": nav_text,
                "source_text": summary or key_text,
                "block_type": None,
                "start_line": node.get("start_line"),
                "end_line": node.get("end_line"),
            })

        seen_block_ids: set[str] = set()

        for block in content_blocks:

            if not isinstance(block, dict):
                raise ValueError(
                    f"Node {node_id!r}: every block must be an object."
                )

            block_id = block.get("block_id")

            if not isinstance(block_id, str) or not block_id.strip():
                raise ValueError(
                    f"Content block in node {node_id!r} has no block_id."
                )

            if block_id in seen_block_ids:
                raise ValueError(
                    f"Duplicate block_id {block_id!r} "
                    f"in node {node_id!r}."
                )

            seen_block_ids.add(block_id)

            source_text = str(block.get("text") or "").strip()

            if not source_text:
                continue

            searchable_text = "\n".join(
                part
                for part in (
                    f"Document: {doc_title}",
                    f"Section: {title}",
                    f"Section summary: {summary}",
                    f"Key concepts: {key_text}",
                    f"Content: {source_text}",
                )
                if part.strip()
            )

            append_record({
                "record_id": _record_id(
                    document_id,
                    "evidence",
                    node_id,
                    block_id,
                ),
                "document_id": document_id,
                "record_type": "evidence",
                "node_id": node_id,
                "parent_id": parent_id,
                "node_title": title,
                "section_path": list(section_path),
                "content_block_id": block_id,
                "text": searchable_text,
                "source_text": source_text,
                "block_type": block.get("type"),
                "start_line": block.get("start_line"),
                "end_line": block.get("end_line"),
            })

    # -------------------------------------------------------------
    # Preamble evidence
    # -------------------------------------------------------------

    seen_preamble_ids: set[str] = set()

    for block in preamble:

        if not isinstance(block, dict):
            raise ValueError(
                "Every preamble block must be an object."
            )

        block_id = block.get("block_id")

        if not isinstance(block_id, str) or not block_id.strip():
            raise ValueError(
                "Preamble content block has no block_id."
            )

        if block_id in seen_preamble_ids:
            raise ValueError(
                f"Duplicate preamble block_id: {block_id}"
            )

        seen_preamble_ids.add(block_id)

        source_text = str(block.get("text") or "").strip()

        if not source_text:
            continue

        append_record({
            "record_id": _record_id(
                document_id,
                "evidence",
                "__preamble__",
                block_id,
            ),
            "document_id": document_id,
            "record_type": "evidence",
            "node_id": "__preamble__",
            "parent_id": None,
            "node_title": "Document preamble",
            "section_path": ["Document preamble"],
            "content_block_id": block_id,
            "text": (
                f"Document: {doc_title}\n"
                f"Section: Document preamble\n"
                f"Content: {source_text}"
            ),
            "source_text": source_text,
            "block_type": block.get("type"),
            "start_line": block.get("start_line"),
            "end_line": block.get("end_line"),
        })

    return records


# ---------------------------------------------------------------------
# SQLite utilities and schema
# ---------------------------------------------------------------------


def open_db(db_path: Path) -> sqlite3.Connection:
    """Open a retrieval database without changing its schema."""

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 30000")

    return conn


def _create_current_tables(conn: sqlite3.Connection) -> None:

    conn.execute("""
        CREATE TABLE records (
            record_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            record_type TEXT NOT NULL,
            node_id TEXT NOT NULL,
            parent_id TEXT,
            node_title TEXT NOT NULL,
            section_path TEXT NOT NULL,
            content_block_id TEXT,
            searchable_text TEXT NOT NULL,
            source_text TEXT NOT NULL,
            block_type TEXT,
            start_line INTEGER,
            end_line INTEGER,
            embedding BLOB NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            doc_name TEXT NOT NULL,
            doc_title TEXT NOT NULL,
            structure_path TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    conn.execute(
        "CREATE INDEX idx_records_document_id ON records(document_id)"
    )


def _table_names(conn: sqlite3.Connection) -> set[str]:

    return {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
        if not row["name"].startswith("sqlite_")
    }


def _columns(
    conn: sqlite3.Connection,
    table: str,
) -> set[str]:

    return {
        row["name"]
        for row in conn.execute(
            f'PRAGMA table_info("{table}")'
        )
    }


def _metadata(conn: sqlite3.Connection) -> dict[str, str]:

    return {
        row["key"]: row["value"]
        for row in conn.execute(
            "SELECT key, value FROM index_metadata"
        )
    }


def _has_document_id_index(
    conn: sqlite3.Connection,
) -> bool:

    return conn.execute("""
        SELECT 1
        FROM sqlite_master
        WHERE type = 'index'
        AND name = 'idx_records_document_id'
    """).fetchone() is not None


def _validate_current_schema(
    conn: sqlite3.Connection,
) -> None:

    tables = _table_names(conn)

    required_tables = {
        "records",
        "documents",
        "index_metadata",
    }

    if not required_tables <= tables:
        raise ValueError(
            "Retrieval schema metadata is current but one or more "
            "required tables are missing. Preserve the database "
            "and rebuild it."
        )

    if not set(RECORD_COLUMNS) <= _columns(conn, "records"):
        raise ValueError(
            "Current retrieval records table has incompatible columns."
        )

    if not set(DOCUMENT_COLUMNS) <= _columns(conn, "documents"):
        raise ValueError(
            "Current retrieval documents table has incompatible columns."
        )

    if not set(METADATA_COLUMNS) <= _columns(conn, "index_metadata"):
        raise ValueError(
            "Current retrieval metadata table has incompatible columns."
        )


def list_indexed_documents(db_path: Path) -> list[dict[str, Any]]:
    """Return the current document registry without loading the embedding model."""
    db_path = Path(db_path)
    if not db_path.is_file():
        raise FileNotFoundError(f"Index not found: {db_path}. Build it first.")

    conn = open_db(db_path)
    try:
        conn.execute("BEGIN")
        _validate_current_schema(conn)
        version = _metadata(conn).get("schema_version")
        if version != str(SCHEMA_VERSION):
            raise ValueError(
                f"Document routing requires schema version {SCHEMA_VERSION}; "
                f"found {version!r}. Run search_index() once to migrate it."
            )
        return [
            dict(row)
            for row in conn.execute(
                """
                SELECT document_id, doc_name, doc_title, structure_path,
                       record_count, indexed_at
                FROM documents
                ORDER BY doc_name, document_id
                """
            )
        ]
    finally:
        conn.rollback()
        conn.close()


# ---------------------------------------------------------------------
# Legacy structure discovery and validation
# ---------------------------------------------------------------------


def _legacy_structure_path(
    db_path: Path,
    document_id: str,
    supplied_structure_path: Path | None,
) -> Path | None:

    candidates: list[Path] = []

    if supplied_structure_path is not None:
        candidates.append(supplied_structure_path.resolve())

    app_dir = db_path.resolve().parent.parent.parent

    candidates.extend((
        app_dir / "structure.json",
        app_dir / "structure.summarized.json",
        app_dir / "storage" / "documents" / document_id / "structure.json",
        app_dir / "documents" / document_id / "structure.json",
    ))

    seen: set[Path] = set()

    for candidate in candidates:

        if candidate in seen or not candidate.is_file():
            continue

        seen.add(candidate)

        try:
            tree = json.loads(
                candidate.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue

        if (
            isinstance(tree, dict)
            and tree.get("document_id") == document_id
        ):
            return candidate

    return None


def _validate_structure_path(
    path: Path | None,
    document_id: str,
) -> Path | None:
    """Return the path only if it contains the expected document structure."""

    if path is None:
        return None

    try:
        resolved_path = Path(path).resolve()

        if not resolved_path.is_file():
            return None

        tree = json.loads(
            resolved_path.read_text(encoding="utf-8")
        )

        if not isinstance(tree, dict):
            return None

        if tree.get("document_id") != document_id:
            return None

        return resolved_path

    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------
# Schema v1 -> v2 migration
# ---------------------------------------------------------------------


def _migrate_v1(
    conn: sqlite3.Connection,
    db_path: Path,
    supplied_structure_path: Path | None,
) -> None:
    """Upgrade the original single-document v1 table without losing records."""

    tables = _table_names(conn)

    if not {"records", "index_metadata"} <= tables:
        raise ValueError(
            "Cannot migrate schema version 1: records or index_metadata "
            "is missing. Preserve the database and explicitly rebuild "
            "the index from each source structure file."
        )

    required_record_columns = set(RECORD_COLUMNS)

    if not required_record_columns <= _columns(conn, "records"):
        raise ValueError(
            "Cannot safely migrate schema version 1: the records table "
            "does not have the expected v1 columns. Preserve the database "
            "and rebuild into a new database from source structure files."
        )

    if not set(METADATA_COLUMNS) <= _columns(conn, "index_metadata"):
        raise ValueError(
            "Cannot safely migrate schema version 1: index_metadata "
            "has incompatible columns."
        )

    metadata = _metadata(conn)

    legacy_ids = {
        row["document_id"]
        for row in conn.execute(
            "SELECT DISTINCT document_id FROM records"
        )
    }

    if None in legacy_ids or "" in legacy_ids:
        raise ValueError(
            "Cannot migrate schema version 1: a record has no document_id. "
            "Rebuild from source structure files into a new database."
        )

    metadata_document_id = metadata.get("document_id")

    if not metadata_document_id:
        raise ValueError(
            "Cannot migrate schema version 1: corpus metadata has no "
            "document_id. Rebuild from source structure files into "
            "a new database."
        )

    legacy_ids.add(metadata_document_id)

    if len(legacy_ids) != 1:
        raise ValueError(
            "Cannot safely migrate schema version 1: its records refer "
            "to multiple documents even though v1 only stored one "
            "document's metadata. Preserve this database and rebuild "
            "into a new multi-document index from source structure files."
        )

    document_id = next(iter(legacy_ids))

    model_name = metadata.get("model_name")

    try:
        dimension = int(metadata["embedding_dimension"])
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Cannot migrate schema version 1: embedding model metadata "
            "is incomplete. Preserve the database and rebuild from "
            "source files."
        ) from error

    if not model_name or dimension <= 0:
        raise ValueError(
            "Cannot migrate schema version 1: embedding model metadata "
            "is invalid. Preserve the database and rebuild from "
            "source files."
        )

    rows = conn.execute(
        "SELECT * FROM records"
    ).fetchall()

    migrated: list[tuple[Any, ...]] = []
    new_ids: set[str] = set()

    expected_bytes = (
        dimension * np.dtype(np.float32).itemsize
    )

    for row in rows:

        row_document_id = row["document_id"]
        record_type = row["record_type"]
        node_id = row["node_id"]
        block_id = row["content_block_id"]
        embedding = row["embedding"]

        valid_identity = (
            row_document_id == document_id
            and record_type in {"navigation", "evidence"}
            and isinstance(node_id, str)
            and bool(node_id.strip())
            and (
                (
                    record_type == "evidence"
                    and isinstance(block_id, str)
                    and bool(block_id.strip())
                )
                or (
                    record_type == "navigation"
                    and block_id is None
                )
            )
        )

        if not valid_identity:
            raise ValueError(
                "Cannot safely migrate schema version 1: a record is "
                "missing valid ownership or node/block identity. "
                "Preserve the database and rebuild from source "
                "structure files."
            )

        if not isinstance(
            embedding,
            (bytes, bytearray, memoryview),
        ):
            raise ValueError(
                "Cannot safely migrate schema version 1: a stored "
                "embedding is not a valid byte string."
            )

        if len(embedding) != expected_bytes:
            raise ValueError(
                "Cannot safely migrate schema version 1: stored "
                "embedding bytes do not match the recorded dimension."
            )

        vector = np.frombuffer(
            embedding,
            dtype=np.float32,
        )

        if not np.isfinite(vector).all():
            raise ValueError(
                "Cannot safely migrate schema version 1: a stored "
                "embedding contains non-finite values."
            )

        try:
            section_path = json.loads(row["section_path"])
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "Cannot safely migrate schema version 1: a record has "
                "an invalid section_path."
            ) from error

        if (
            not isinstance(section_path, list)
            or not all(isinstance(item, str) for item in section_path)
        ):
            raise ValueError(
                "Cannot safely migrate schema version 1: a record has "
                "an invalid section_path."
            )

        record_id = _record_id(
            document_id,
            record_type,
            node_id,
            block_id,
        )

        if record_id in new_ids:
            raise ValueError(
                "Cannot safely migrate schema version 1: multiple rows "
                "map to the same composite record identity. Preserve "
                "the database and rebuild from source structure files."
            )

        new_ids.add(record_id)

        migrated.append((
            record_id,
            document_id,
            record_type,
            node_id,
            row["parent_id"],
            row["node_title"],
            row["section_path"],
            block_id,
            row["searchable_text"],
            row["source_text"],
            row["block_type"],
            row["start_line"],
            row["end_line"],
            embedding,
        ))

    # -------------------------------------------------------------
    # Read the old document registry, if one exists
    # -------------------------------------------------------------

    old_documents = []

    if "documents" in tables:

        if not set(DOCUMENT_COLUMNS) <= _columns(
            conn,
            "documents",
        ):
            raise ValueError(
                "Cannot safely migrate schema version 1: an incompatible "
                "documents table already exists. Preserve the database "
                "and rebuild into a new database."
            )

        old_documents = conn.execute(
            "SELECT * FROM documents"
        ).fetchall()

        unexpected_document_ids = {
            row["document_id"]
            for row in old_documents
            if row["document_id"] != document_id
        }

        if unexpected_document_ids:
            unexpected = ", ".join(
                sorted(unexpected_document_ids)
            )

            raise ValueError(
                "Cannot safely migrate schema version 1: the legacy "
                "documents registry contains additional document "
                f"entries ({unexpected}). Preserve this database "
                "and rebuild from all source structure files into "
                "a new database."
            )

    # -------------------------------------------------------------
    # Recover document metadata and structure path
    # -------------------------------------------------------------

    structure_path = _legacy_structure_path(
        db_path,
        document_id,
        supplied_structure_path,
    )

    doc_name = metadata.get("doc_name", "")
    doc_title = metadata.get("doc_title", "")

    if structure_path is not None:

        try:
            tree = json.loads(
                structure_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, json.JSONDecodeError):
            tree = {}

        if isinstance(tree, dict):
            doc_name = str(
                tree.get("doc_name") or doc_name
            )
            doc_title = str(
                tree.get("doc_title") or doc_title
            )

    for old_document in old_documents:

        if old_document["document_id"] != document_id:
            continue

        doc_name = str(
            old_document["doc_name"] or doc_name
        )

        doc_title = str(
            old_document["doc_title"] or doc_title
        )

        if structure_path is None:

            legacy_path = old_document["structure_path"]

            if legacy_path:
                structure_path = _validate_structure_path(
                    Path(legacy_path),
                    document_id,
                )

    # -------------------------------------------------------------
    # Replace old schema with v2 schema
    # -------------------------------------------------------------

    conn.execute("DROP TABLE records")

    if "documents" in tables:
        conn.execute("DROP TABLE documents")

    conn.execute("DROP TABLE index_metadata")

    _create_current_tables(conn)

    conn.executemany(
        """
        INSERT INTO records (
            record_id,
            document_id,
            record_type,
            node_id,
            parent_id,
            node_title,
            section_path,
            content_block_id,
            searchable_text,
            source_text,
            block_type,
            start_line,
            end_line,
            embedding
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        migrated,
    )

    conn.execute(
        """
        INSERT INTO documents (
            document_id,
            doc_name,
            doc_title,
            structure_path,
            record_count
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            document_id,
            doc_name,
            doc_title,
            (
                str(structure_path)
                if structure_path is not None
                else ""
            ),
            len(migrated),
        ),
    )

    retained_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {
            "schema_version",
            "document_id",
            "doc_name",
            "doc_title",
            "record_count",
            "migration_warning",
        }
    }

    retained_metadata.update({
        "schema_version": str(SCHEMA_VERSION),
        "model_name": model_name,
        "embedding_dimension": str(dimension),
    })

    if structure_path is None:
        retained_metadata["migration_warning"] = (
            f"Legacy document {document_id} has no known structure_path. "
            "Reindex it from its source structure.json before "
            "context expansion."
        )

    conn.executemany(
        """
        INSERT INTO index_metadata (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE
        SET value = excluded.value
        """,
        list(retained_metadata.items()),
    )


# ---------------------------------------------------------------------
# Schema initialization and validation
# ---------------------------------------------------------------------


def _ensure_schema(
    conn: sqlite3.Connection,
    db_path: Path,
    supplied_structure_path: Path | None = None,
) -> None:

    tables = _table_names(conn)

    if not tables:

        _create_current_tables(conn)

        conn.execute(
            """
            INSERT INTO index_metadata (key, value)
            VALUES (?, ?)
            """,
            (
                "schema_version",
                str(SCHEMA_VERSION),
            ),
        )

        return

    if "index_metadata" not in tables:
        raise ValueError(
            "Retrieval database has no index_metadata table; refusing "
            "to interpret an unknown schema as the current format. "
            "Preserve it and rebuild into a new database."
        )

    if not set(METADATA_COLUMNS) <= _columns(
        conn,
        "index_metadata",
    ):
        raise ValueError(
            "Retrieval database has incompatible index_metadata columns."
        )

    metadata = _metadata(conn)
    version = metadata.get("schema_version")

    if version == "1":

        _migrate_v1(
            conn,
            db_path,
            supplied_structure_path,
        )

        return

    if version != str(SCHEMA_VERSION):
        raise ValueError(
            f"Unsupported retrieval schema version {version!r}; "
            f"expected {SCHEMA_VERSION}. Preserve the database and "
            "use an explicit migration or rebuild into a new database."
        )

    _validate_current_schema(conn)

    if not _has_document_id_index(conn):
        conn.execute(
            """
            CREATE INDEX idx_records_document_id
            ON records(document_id)
            """
        )


def _validate_vector_metadata(
    metadata: dict[str, str],
    model_name: str,
    dimension: int,
) -> None:

    existing_model = metadata.get("model_name")

    if existing_model and existing_model != model_name:
        raise ValueError(
            f"Embedding model mismatch. Existing index uses "
            f"{existing_model}, but current model is {model_name}. "
            "Rebuild the corpus with one embedding model."
        )

    existing_dimension = metadata.get("embedding_dimension")

    if existing_dimension:

        try:
            existing_dimension_value = int(existing_dimension)
        except ValueError as error:
            raise ValueError(
                "Existing embedding dimension metadata is invalid."
            ) from error

        if existing_dimension_value <= 0:
            raise ValueError(
                "Existing embedding dimension must be positive."
            )

        if existing_dimension_value != dimension:
            raise ValueError(
                "Embedding dimension mismatch. All documents in one "
                "index must use the same embedding dimension."
            )


# ---------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------


def build_index(
    structure_path: Path,
    db_path: Path,
    batch_size: int = 32,
) -> None:

    structure_path = Path(structure_path).resolve()
    db_path = Path(db_path)

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be a positive integer."
        )

    tree = json.loads(
        structure_path.read_text(encoding="utf-8")
    )

    records = make_records(tree)

    document_id = tree["document_id"]

    doc_name = str(
        tree.get("doc_name") or ""
    )

    doc_title = str(
        tree.get("doc_title") or ""
    )

    print(
        f"Preparing {len(records)} records "
        f"for document: {document_id}"
    )

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
    )

    model_dimension = model.get_sentence_embedding_dimension()

    if (
        not isinstance(model_dimension, int)
        or model_dimension <= 0
    ):
        raise ValueError(
            "Embedding model did not report a valid dimension."
        )

    # Generate embeddings before opening the write transaction.
    # Expensive model inference should not hold a database lock.

    if records:

        vectors = np.asarray(
            model.encode(
                [record["text"] for record in records],
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

        if (
            vectors.ndim != 2
            or vectors.shape != (
                len(records),
                model_dimension,
            )
        ):
            raise ValueError(
                "Embedding model returned an unexpected number "
                "of vectors or dimensions."
            )

        if not np.isfinite(vectors).all():
            raise ValueError(
                "Embedding model returned non-finite vector values."
            )

    else:
        vectors = np.empty(
            (0, model_dimension),
            dtype=np.float32,
        )

    conn = open_db(db_path)

    try:

        conn.execute("BEGIN IMMEDIATE")

        _ensure_schema(
            conn,
            db_path,
            structure_path,
        )

        metadata = _metadata(conn)

        _validate_vector_metadata(
            metadata,
            MODEL_NAME,
            model_dimension,
        )

        # Replace only this document's records.
        # Other documents remain untouched.

        conn.execute(
            """
            DELETE FROM records
            WHERE document_id = ?
            """,
            (document_id,),
        )

        rows = [
            (
                record["record_id"],
                record["document_id"],
                record["record_type"],
                record["node_id"],
                record["parent_id"],
                record["node_title"],
                json.dumps(
                    record["section_path"],
                    ensure_ascii=False,
                ),
                record["content_block_id"],
                record["text"],
                record["source_text"],
                record.get("block_type"),
                record.get("start_line"),
                record.get("end_line"),
                vector.tobytes(),
            )
            for record, vector in zip(records, vectors)
        ]

        conn.executemany(
            """
            INSERT INTO records (
                record_id,
                document_id,
                record_type,
                node_id,
                parent_id,
                node_title,
                section_path,
                content_block_id,
                searchable_text,
                source_text,
                block_type,
                start_line,
                end_line,
                embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        # Register or update this document.

        conn.execute(
            """
            INSERT INTO documents (
                document_id,
                doc_name,
                doc_title,
                structure_path,
                record_count,
                indexed_at
            )
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)

            ON CONFLICT(document_id) DO UPDATE SET
                doc_name = excluded.doc_name,
                doc_title = excluded.doc_title,
                structure_path = excluded.structure_path,
                record_count = excluded.record_count,
                indexed_at = CURRENT_TIMESTAMP
            """,
            (
                document_id,
                doc_name,
                doc_title,
                str(structure_path),
                len(records),
            ),
        )

        # Update corpus-wide embedding metadata.

        metadata_updates = {
            "schema_version": str(SCHEMA_VERSION),
            "model_name": MODEL_NAME,
            "embedding_dimension": str(model_dimension),
        }

        conn.executemany(
            """
            INSERT INTO index_metadata (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE
            SET value = excluded.value
            """,
            list(metadata_updates.items()),
        )

        # Clear a migration warning after this document has been reindexed.

        warning = metadata.get("migration_warning")

        if warning and f"document {document_id} " in warning:
            conn.execute(
                """
                DELETE FROM index_metadata
                WHERE key = 'migration_warning'
                """
            )

        conn.commit()

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

    print(
        f"Indexed {len(records)} records "
        f"for document: {document_id}"
    )

    print(f"Index saved to: {db_path}")


# ---------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = 60,
) -> dict[str, float]:
    """Fuse rankings without comparing incompatible raw scores."""

    if rrf_k <= 0:
        raise ValueError("rrf_k must be positive.")

    scores: dict[str, float] = {}

    for ranked in ranked_lists:

        for rank, record_id in enumerate(
            ranked,
            start=1,
        ):

            scores[record_id] = (
                scores.get(record_id, 0.0)
                + 1.0 / (rrf_k + rank)
            )

    return scores


# ---------------------------------------------------------------------
# Hybrid retrieval
# ---------------------------------------------------------------------


def search_index(
    query: str,
    db_path: Path,
    *,
    top_k: int = 8,
    candidate_k: int = 30,
    document_ids: Iterable[str] | None = None,
) -> dict[str, Any]:

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty.")

    for name, value in (
        ("top_k", top_k),
        ("candidate_k", candidate_k),
    ):

        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
        ):
            raise ValueError(
                f"{name} must be a positive integer."
            )

    selected_document_ids: list[str] | None = None
    if document_ids is not None:
        if isinstance(document_ids, (str, bytes)):
            raise ValueError("document_ids must be a collection of document IDs.")
        try:
            supplied_document_ids = list(document_ids)
        except TypeError as error:
            raise ValueError(
                "document_ids must be a collection of document IDs."
            ) from error
        for index, document_id in enumerate(supplied_document_ids):
            if (
                not isinstance(document_id, str)
                or not document_id
                or document_id != document_id.strip()
            ):
                raise ValueError(
                    f"document_ids[{index}] must be a non-empty, trimmed string."
                )
        selected_document_ids = list(dict.fromkeys(supplied_document_ids))
        if not selected_document_ids:
            raise ValueError("document_ids cannot be empty when supplied.")

    db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(
            f"Index not found: {db_path}. Build it first."
        )

    conn = open_db(db_path)

    needs_schema_write = False

    try:

        # Start with a read transaction.
        # Search should not acquire a write lock unnecessarily.

        conn.execute("BEGIN")

        tables = _table_names(conn)

        if not tables:

            needs_schema_write = True

        elif "index_metadata" not in tables:

            _ensure_schema(
                conn,
                db_path,
            )

        else:

            if not set(METADATA_COLUMNS) <= _columns(
                conn,
                "index_metadata",
            ):
                raise ValueError(
                    "Retrieval database has incompatible metadata columns."
                )

            version = _metadata(conn).get("schema_version")

            if version == "1":

                needs_schema_write = True

            elif (
                version == str(SCHEMA_VERSION)
                and not _has_document_id_index(conn)
            ):

                needs_schema_write = True

            else:

                _ensure_schema(
                    conn,
                    db_path,
                )

        # Migration and index creation require a write transaction.
        # Perform them separately, then start a fresh read snapshot.

        if needs_schema_write:

            conn.rollback()
            conn.close()

            conn = open_db(db_path)
            conn.execute("BEGIN IMMEDIATE")

            _ensure_schema(
                conn,
                db_path,
            )

            conn.commit()
            conn.close()

            conn = open_db(db_path)
            conn.execute("BEGIN")

            _validate_current_schema(conn)

        # Read all corpus data from one consistent snapshot.

        rows = conn.execute(
            "SELECT * FROM records"
        ).fetchall()

        metadata = _metadata(conn)

        documents = [
            dict(row)
            for row in conn.execute(
                """
                SELECT
                    document_id,
                    doc_name,
                    doc_title,
                    structure_path,
                    record_count,
                    indexed_at
                FROM documents
                ORDER BY doc_name, document_id
                """
            ).fetchall()
        ]

        if selected_document_ids is not None:
            registered_ids = {
                document["document_id"] for document in documents
            }
            unknown_ids = set(selected_document_ids) - registered_ids
            if unknown_ids:
                raise ValueError(
                    "Unknown document_id(s): " + ", ".join(sorted(unknown_ids))
                )
            selected_ids = set(selected_document_ids)
            rows = [
                row for row in rows
                if row["document_id"] in selected_ids
            ]
            documents = [
                document for document in documents
                if document["document_id"] in selected_ids
            ]

        conn.commit()

    except Exception:

        conn.rollback()
        raise

    finally:

        conn.close()

    if not rows:
        raise ValueError(
            "The retrieval index contains no searchable records. "
            "Indexed documents with zero records remain registered."
        )

    # -------------------------------------------------------------
    # Validate corpus metadata
    # -------------------------------------------------------------

    try:
        expected_dim = int(
            metadata["embedding_dimension"]
        )
    except (KeyError, ValueError) as error:
        raise ValueError(
            "Index embedding dimension metadata is missing or invalid."
        ) from error

    if expected_dim <= 0:
        raise ValueError(
            "Index embedding dimension must be positive."
        )

    model_name = metadata.get("model_name")

    if not model_name:
        raise ValueError(
            "Index embedding model metadata is missing."
        )

    document_by_id = {
        document["document_id"]: document
        for document in documents
    }

    records_by_document: dict[str, int] = {}

    vectors_list: list[np.ndarray] = []

    expected_bytes = (
        expected_dim * np.dtype(np.float32).itemsize
    )

    # -------------------------------------------------------------
    # Validate records and stored embeddings
    # -------------------------------------------------------------

    for row in rows:

        record_id = row["record_id"]
        document_id = row["document_id"]

        if document_id not in document_by_id:
            raise ValueError(
                f"Record {record_id!r} references unregistered "
                f"document {document_id!r}. Reindex or rebuild "
                "the corpus."
            )

        if row["record_type"] not in {
            "navigation",
            "evidence",
        }:
            raise ValueError(
                f"Record {record_id!r} has an unknown record_type."
            )

        embedding = row["embedding"]

        if not isinstance(
            embedding,
            (bytes, bytearray, memoryview),
        ):
            raise ValueError(
                f"Stored embedding for record {record_id!r} "
                "is not a byte string."
            )

        if len(embedding) != expected_bytes:
            raise ValueError(
                f"Stored embedding for record {record_id!r} "
                f"has invalid byte length: expected {expected_bytes}, "
                f"got {len(embedding)}."
            )

        vector = np.frombuffer(
            embedding,
            dtype=np.float32,
        )

        if vector.size != expected_dim:
            raise ValueError(
                f"Stored embedding for record {record_id!r} "
                f"has dimension {vector.size}; expected {expected_dim}."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                f"Stored embedding for record {record_id!r} "
                "contains non-finite values."
            )

        vectors_list.append(vector)

        records_by_document[document_id] = (
            records_by_document.get(document_id, 0) + 1
        )

    # Validate document registry counts.

    for document in documents:

        actual_count = records_by_document.get(
            document["document_id"],
            0,
        )

        if actual_count != document["record_count"]:
            raise ValueError(
                f"Document registry count for "
                f"{document['document_id']!r} is "
                f"{document['record_count']}, but the records "
                f"table contains {actual_count}."
            )

    vectors = np.vstack(vectors_list)

    # -------------------------------------------------------------
    # Dense retrieval
    # -------------------------------------------------------------

    model = SentenceTransformer(
        model_name,
        device="cpu",
    )

    query_vector = np.asarray(
        model.encode(
            [QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0],
        dtype=np.float32,
    )

    if query_vector.shape != (expected_dim,):
        raise ValueError(
            "Query embedding dimension does not match "
            "the indexed corpus."
        )

    if not np.isfinite(query_vector).all():
        raise ValueError(
            "Query embedding contains non-finite values."
        )

    dense_scores = vectors @ query_vector

    dense_order = np.argsort(
        -dense_scores,
        kind="stable",
    )[:candidate_k]

    dense_ranked_ids = [
        rows[int(index)]["record_id"]
        for index in dense_order
    ]

    # -------------------------------------------------------------
    # BM25 retrieval
    # -------------------------------------------------------------

    corpus_tokens = [
        tokenize(row["searchable_text"])
        for row in rows
    ]

    query_tokens = tokenize(query)

    bm25_scores = np.zeros(
        len(rows),
        dtype=np.float64,
    )

    bm25_ranked_ids: list[str] = []

    tokenized_indexes = [
        index
        for index, tokens in enumerate(corpus_tokens)
        if tokens
    ]

    if query_tokens and tokenized_indexes:

        bm25 = BM25Okapi([
            corpus_tokens[index]
            for index in tokenized_indexes
        ])

        tokenized_scores = bm25.get_scores(
            query_tokens
        )

        for index, score in zip(
            tokenized_indexes,
            tokenized_scores,
        ):
            bm25_scores[index] = float(score)

        # Precompute token sets once instead of repeatedly
        # constructing them during matching.

        token_sets = [
            set(tokens)
            for tokens in corpus_tokens
        ]

        # Include every record containing at least one query token.
        # Do not require a positive BM25 score: depending on corpus
        # statistics, a matching record can receive zero or negative
        # scores.

        matching_indexes = [
            index
            for index in tokenized_indexes
            if any(
                token in token_sets[index]
                for token in query_tokens
            )
        ]

        # Score descending, then original corpus position.
        # This gives deterministic tie-breaking.

        matching_indexes.sort(
            key=lambda index: (
                -bm25_scores[index],
                index,
            )
        )

        bm25_ranked_ids = [
            rows[index]["record_id"]
            for index in matching_indexes[:candidate_k]
        ]

    # -------------------------------------------------------------
    # Reciprocal Rank Fusion
    # -------------------------------------------------------------

    fused_scores = reciprocal_rank_fusion([
        dense_ranked_ids,
        bm25_ranked_ids,
    ])

    row_by_id = {
        row["record_id"]: row
        for row in rows
    }

    dense_score_by_id = {
        row["record_id"]: float(dense_scores[index])
        for index, row in enumerate(rows)
    }

    bm25_score_by_id = {
        row["record_id"]: float(bm25_scores[index])
        for index, row in enumerate(rows)
    }

    ranked_ids = sorted(
        fused_scores,
        key=lambda record_id: (
            -fused_scores[record_id],
            record_id,
        ),
    )

    # -------------------------------------------------------------
    # Format results
    # -------------------------------------------------------------

    evidence: list[dict[str, Any]] = []
    navigation: list[dict[str, Any]] = []

    for record_id in ranked_ids:

        row = row_by_id[record_id]

        document = document_by_id[
            row["document_id"]
        ]

        source_text = row["source_text"]

        item = {
            "record_id": row["record_id"],
            "document_id": row["document_id"],
            "doc_name": document["doc_name"],
            "doc_title": document["doc_title"],
            "record_type": row["record_type"],
            "node_id": row["node_id"],
            "node_title": row["node_title"],
            "section_path": json.loads(
                row["section_path"]
            ),
            "content_block_id": row["content_block_id"],
            "text": source_text,
            "source_text": source_text,
            "block_type": row["block_type"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "fusion_score": round(
                float(fused_scores[record_id]),
                6,
            ),
            "dense_score": round(
                dense_score_by_id[record_id],
                6,
            ),
            "bm25_score": round(
                bm25_score_by_id[record_id],
                6,
            ),
        }

        if row["record_type"] == "evidence":
            evidence.append(item)

        else:
            navigation.append(item)

    return {
        "query": query,
        "retrieval_status": (
            "found"
            if evidence[:top_k]
            else "not_found"
        ),
        "status_note": (
            "Status only indicates whether source blocks were retrieved. "
            "It does not establish that they fully answer the question."
        ),
        "evidence": evidence[:top_k],
        "navigation_hits": navigation[:top_k],
        "index_metadata": metadata,
        "documents": documents,
    }


def get_subtree_evidence_records(
    db_path: Path,
    document_id: str,
    node_ids: Iterable[str],
) -> list[dict[str, Any]]:
    """Return original evidence records for selected document nodes.

    Records retain the supplied node order and are ordered by source position
    within each node, with record ID as the final deterministic tie-breaker.
    Ranking scores are ``None`` because this lookup does not perform retrieval.
    """

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")

    if document_id != document_id.strip():
        raise ValueError(
            "document_id must not contain leading or trailing whitespace."
        )

    if isinstance(node_ids, (str, bytes)):
        raise ValueError("node_ids must be a collection of node ID strings.")

    try:
        supplied_node_ids = list(node_ids)
    except TypeError as error:
        raise ValueError(
            "node_ids must be a collection of node ID strings."
        ) from error

    for index, node_id in enumerate(supplied_node_ids):
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError(
                f"node_ids[{index}] must be a non-empty string."
            )
        if node_id != node_id.strip():
            raise ValueError(
                f"node_ids[{index}] must not contain leading or trailing "
                "whitespace."
            )

    # Preamble records use a reserved node ID and are not part of a subtree.
    unique_node_ids = list(
        dict.fromkeys(
            node_id
            for node_id in supplied_node_ids
            if node_id != "__preamble__"
        )
    )
    if not unique_node_ids:
        return []

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Index not found: {db_path}. Build it first."
        )

    conn = open_db(db_path)
    try:
        conn.execute("BEGIN")
        _validate_current_schema(conn)

        schema_version = _metadata(conn).get("schema_version")
        if schema_version != str(SCHEMA_VERSION):
            raise ValueError(
                "Subtree evidence lookup requires retrieval schema "
                f"version {SCHEMA_VERSION}; found {schema_version!r}. "
                "Use search_index() to initialize or migrate the index."
            )

        requested_nodes = ", ".join(
            "(?, ?)" for _ in unique_node_ids
        )
        requested_node_parameters = [
            parameter
            for order, node_id in enumerate(unique_node_ids)
            for parameter in (node_id, order)
        ]
        rows = conn.execute(
            f"""
            WITH requested_nodes(node_id, node_order) AS (
                VALUES {requested_nodes}
            )
            SELECT
                r.record_id,
                r.document_id,
                r.record_type,
                r.node_id,
                r.node_title,
                r.section_path,
                r.content_block_id,
                r.source_text,
                r.block_type,
                r.start_line,
                r.end_line,
                d.doc_name,
                d.doc_title
            FROM records AS r
            LEFT JOIN documents AS d
                ON d.document_id = r.document_id
            JOIN requested_nodes
                ON requested_nodes.node_id = r.node_id
            WHERE r.document_id = ?
                AND r.record_type = 'evidence'
                AND r.node_id != ?
            ORDER BY
                requested_nodes.node_order,
                r.start_line,
                r.end_line,
                r.record_id
            """,
            (
                *requested_node_parameters,
                document_id,
                "__preamble__",
            ),
        ).fetchall()

        results: list[dict[str, Any]] = []
        for row in rows:
            if row["doc_name"] is None or row["doc_title"] is None:
                raise ValueError(
                    f"Record {row['record_id']!r} references unregistered "
                    f"document {document_id!r}."
                )

            source_text = row["source_text"]
            results.append({
                "record_id": row["record_id"],
                "document_id": row["document_id"],
                "doc_name": row["doc_name"],
                "doc_title": row["doc_title"],
                "record_type": row["record_type"],
                "node_id": row["node_id"],
                "node_title": row["node_title"],
                "section_path": json.loads(row["section_path"]),
                "content_block_id": row["content_block_id"],
                "text": source_text,
                "source_text": source_text,
                "block_type": row["block_type"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "fusion_score": None,
                "dense_score": None,
                "bm25_score": None,
            })

        conn.commit()
        return results

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Local hybrid retrieval over a PageIndex-style document tree."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Build/rebuild the local index.",
    )

    index_parser.add_argument(
        "structure",
        type=Path,
    )

    index_parser.add_argument(
        "--db",
        type=Path,
        default=Path("storage/retrieval/index.sqlite3"),
    )

    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search the local index.",
    )

    search_parser.add_argument(
        "query",
    )

    search_parser.add_argument(
        "--db",
        type=Path,
        default=Path("storage/retrieval/index.sqlite3"),
    )

    search_parser.add_argument(
        "--top-k",
        type=int,
        default=8,
    )

    search_parser.add_argument(
        "--candidate-k",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    if args.command == "index":

        build_index(
            args.structure,
            args.db,
            args.batch_size,
        )

    else:

        result = search_index(
            args.query,
            args.db,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
