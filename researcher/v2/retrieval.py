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
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SCHEMA_VERSION = 1


def walk_tree(
    nodes: list[dict[str, Any]],
    parent_id: str | None = None,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[dict[str, Any], str | None, tuple[str, ...]]]:
    """Yield each node with its parent ID and human-readable section path."""
    for node in nodes:
        title = str(node.get("title") or "Untitled section")
        current_path = path + (title,)
        yield node, parent_id, current_path
        yield from walk_tree(node.get("children") or [], node.get("node_id"), current_path)


def tokenize(text: str) -> list[str]:
    """Simple deterministic tokenizer for BM25; keeps technical tokens like BERT-base."""
    return re.findall(r"[a-zA-Z0-9]+(?:[._/-][a-zA-Z0-9]+)*", text.lower())


def make_records(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Create navigation records and original-evidence records from structure.json."""
    records: list[dict[str, Any]] = []
    document_id = str(tree.get("document_id") or tree.get("doc_name") or "document")
    doc_title = str(tree.get("doc_title") or tree.get("doc_name") or "Untitled document")

    for node, parent_id, section_path in walk_tree(tree.get("structure") or []):
        node_id = str(node.get("node_id") or "")
        title = str(node.get("title") or "Untitled section")
        summary = str(node.get("summary") or "")
        key_items = node.get("key_items") or []
        key_text = "; ".join(str(item) for item in key_items)

        # Navigation records help locate sections, but are never returned as source evidence.
        nav_text = "\n".join(
            part for part in (
                f"Document: {doc_title}",
                f"Section: {title}",
                f"Summary: {summary}",
                f"Key concepts: {key_text}",
            ) if part.strip()
        )
        if summary or key_text:
            records.append({
                "record_id": f"nav:{node_id}",
                "document_id": document_id,
                "record_type": "navigation",
                "node_id": node_id,
                "parent_id": parent_id,
                "node_title": title,
                "section_path": list(section_path),
                "content_block_id": None,
                "text": nav_text,
                "source_text": summary or key_text,
                "start_line": node.get("start_line"),
                "end_line": node.get("end_line"),
            })

        for block in node.get("content_blocks") or []:
            source_text = str(block.get("text") or "").strip()
            if not source_text:
                continue
            block_id = str(block.get("block_id") or "")
            # Add section context to the embedding/search text, but preserve source text separately.
            searchable_text = "\n".join(
                part for part in (
                    f"Document: {doc_title}",
                    f"Section: {title}",
                    f"Section summary: {summary}",
                    f"Key concepts: {key_text}",
                    f"Content: {source_text}",
                ) if part.strip()
            )
            records.append({
                "record_id": f"ev:{node_id}:{block_id}",
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

    # Preamble has no section node. Preserve it as evidence under a synthetic node.
    for block in tree.get("preamble") or []:
        source_text = str(block.get("text") or "").strip()
        if not source_text:
            continue
        block_id = str(block.get("block_id") or "")
        records.append({
            "record_id": f"ev:__preamble__:{block_id}",
            "document_id": document_id,
            "record_type": "evidence",
            "node_id": "__preamble__",
            "parent_id": None,
            "node_title": "Document preamble",
            "section_path": ["Document preamble"],
            "content_block_id": block_id,
            "text": f"Document: {doc_title}\nSection: Document preamble\nContent: {source_text}",
            "source_text": source_text,
            "block_type": block.get("type"),
            "start_line": block.get("start_line"),
            "end_line": block.get("end_line"),
        })

    return records


def open_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS records (
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
        CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def build_index(structure_path: Path, db_path: Path, batch_size: int = 32) -> None:
    tree = json.loads(structure_path.read_text(encoding="utf-8"))
    records = make_records(tree)
    if not records:
        raise ValueError("No summaries or content blocks found in structure.json.")

    print(f"Preparing {len(records)} records...")
    print(f"Loading local embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    texts = [record["text"] for record in records]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    conn = open_db(db_path)
    try:
        # Rebuild is deliberate: structure.json remains the source of truth.
        conn.execute("DELETE FROM records")
        conn.execute("DELETE FROM index_metadata")
        rows = []
        for record, vector in zip(records, vectors):
            rows.append((
                record["record_id"],
                record["document_id"],
                record["record_type"],
                record["node_id"],
                record["parent_id"],
                record["node_title"],
                json.dumps(record["section_path"], ensure_ascii=False),
                record["content_block_id"],
                record["text"],
                record["source_text"],
                record.get("block_type"),
                record.get("start_line"),
                record.get("end_line"),
                vector.tobytes(),
            ))
        conn.executemany("""
            INSERT INTO records (
                record_id, document_id, record_type, node_id, parent_id,
                node_title, section_path, content_block_id, searchable_text,
                source_text, block_type, start_line, end_line, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

        metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "model_name": MODEL_NAME,
            "embedding_dimension": str(vectors.shape[1]),
            "document_id": str(tree.get("document_id") or ""),
            "doc_name": str(tree.get("doc_name") or ""),
            "doc_title": str(tree.get("doc_title") or ""),
            "record_count": str(len(records)),
        }
        conn.executemany(
            "INSERT INTO index_metadata(key, value) VALUES (?, ?)",
            list(metadata.items()),
        )
        conn.commit()
    finally:
        conn.close()

    print(f"Indexed {len(records)} records.")
    print(f"Index saved to: {db_path}")


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = 60,
) -> dict[str, float]:
    """Fuse rankings without pretending dense and BM25 raw scores are comparable."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, record_id in enumerate(ranked, start=1):
            scores[record_id] = scores.get(record_id, 0.0) + 1.0 / (rrf_k + rank)
    return scores


def search_index(
    query: str,
    db_path: Path,
    *,
    top_k: int = 8,
    candidate_k: int = 30,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if not db_path.exists():
        raise FileNotFoundError(f"Index not found: {db_path}. Build it first.")

    conn = open_db(db_path)
    try:
        rows = conn.execute("SELECT * FROM records").fetchall()
        metadata = {
            row["key"]: row["value"]
            for row in conn.execute("SELECT key, value FROM index_metadata").fetchall()
        }
    finally:
        conn.close()

    if not rows:
        raise ValueError("The retrieval index is empty. Build it again.")

    expected_dim = int(metadata.get("embedding_dimension", "0"))
    vectors = np.vstack([
        np.frombuffer(row["embedding"], dtype=np.float32)
        for row in rows
    ])
    if vectors.ndim != 2 or vectors.shape[1] != expected_dim:
        raise ValueError("Stored embedding dimensions do not match index metadata.")

    model_name = metadata.get("model_name", MODEL_NAME)
    model = SentenceTransformer(model_name, device="cpu")
    query_vector = model.encode(
        [QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(np.float32)

    dense_scores = vectors @ query_vector
    dense_order = np.argsort(-dense_scores)[:candidate_k]
    dense_ranked_ids = [rows[int(i)]["record_id"] for i in dense_order]

    corpus_tokens = [tokenize(row["searchable_text"]) for row in rows]
    bm25 = BM25Okapi(corpus_tokens)
    query_tokens = tokenize(query)
    bm25_scores = bm25.get_scores(query_tokens)
    bm25_order = np.argsort(-bm25_scores)[:candidate_k]
    bm25_ranked_ids = [rows[int(i)]["record_id"] for i in bm25_order]

    fused_scores = reciprocal_rank_fusion([dense_ranked_ids, bm25_ranked_ids])
    row_by_id = {row["record_id"]: row for row in rows}
    ranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)

    evidence = []
    navigation = []
    for record_id in ranked_ids:
        row = row_by_id[record_id]
        item = {
            "record_id": row["record_id"],
            "node_id": row["node_id"],
            "node_title": row["node_title"],
            "section_path": json.loads(row["section_path"]),
            "content_block_id": row["content_block_id"],
            "text": row["source_text"],
            "block_type": row["block_type"],
            "start_line": row["start_line"],
            "end_line": row["end_line"],
            "fusion_score": round(float(fused_scores[record_id]), 6),
            "dense_score": round(float(dense_scores[next(
                i for i, candidate in enumerate(rows)
                if candidate["record_id"] == record_id
            )]), 6),
        }
        if row["record_type"] == "evidence":
            evidence.append(item)
        else:
            navigation.append(item)

    # A retrieval hit is not a calibrated confidence estimate or proof of answerability.
    selected_evidence = evidence[:top_k]
    status = "found" if selected_evidence else "not_found"
    return {
        "query": query,
        "retrieval_status": status,
        "status_note": (
            "Status only indicates whether source blocks were retrieved. "
            "It does not establish that they fully answer the question."
        ),
        "evidence": selected_evidence,
        "navigation_hits": navigation[:top_k],
        "index_metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local hybrid retrieval over a PageIndex-style document tree."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build/rebuild the local index.")
    index_parser.add_argument("structure", type=Path)
    index_parser.add_argument(
        "--db",
        type=Path,
        default=Path("storage/retrieval/index.sqlite3"),
    )
    index_parser.add_argument("--batch-size", type=int, default=32)

    search_parser = subparsers.add_parser("search", help="Search the local index.")
    search_parser.add_argument("query")
    search_parser.add_argument(
        "--db",
        type=Path,
        default=Path("storage/retrieval/index.sqlite3"),
    )
    search_parser.add_argument("--top-k", type=int, default=8)
    search_parser.add_argument("--candidate-k", type=int, default=30)

    args = parser.parse_args()
    if args.command == "index":
        build_index(args.structure, args.db, args.batch_size)
    elif args.command == "search":
        result = search_index(
            args.query,
            args.db,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
