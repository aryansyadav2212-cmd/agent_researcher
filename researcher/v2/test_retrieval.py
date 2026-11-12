from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .retrieval import (
    MODEL_NAME,
    SCHEMA_VERSION,
    build_index,
    get_subtree_evidence_records,
    make_records,
    reciprocal_rank_fusion,
    search_index,
)


class FakeSentenceTransformer:
    dimension = 3

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def encode(self, texts, **kwargs):
        return np.tile(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            (len(texts), 1),
        )


def _tree(
    document_id: str,
    text: str,
    *,
    doc_name: str | None = None,
    node_id: str = "same-node",
    block_id: str = "same-block",
    summary: str = "Shared section summary",
) -> dict:
    return {
        "document_id": document_id,
        "doc_name": doc_name or document_id,
        "doc_title": f"Title {doc_name or document_id}",
        "preamble": [],
        "structure": [
            {
                "node_id": node_id,
                "title": "Identical section title",
                "summary": summary,
                "key_items": [],
                "content_blocks": [
                    {
                        "block_id": block_id,
                        "type": "paragraph",
                        "text": text,
                    }
                ],
                "children": [],
            }
        ],
    }


def _write_tree(directory: Path, tree: dict, filename: str = "structure.json") -> Path:
    path = directory / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree), encoding="utf-8")
    return path


def _create_v1_database(
    db_path: Path,
    *,
    node_id: str = "same-node",
    documents: list[tuple[str, str, str, str, int]] | None = None,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.executescript("""
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
            );
            CREATE TABLE index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)
        connection.executemany(
            "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
            [
                ("schema_version", "1"),
                ("document_id", "legacy-doc"),
                ("doc_name", "legacy-doc"),
                ("doc_title", "Title legacy-doc"),
                ("record_count", "1"),
                ("model_name", MODEL_NAME),
                ("embedding_dimension", "3"),
            ],
        )
        connection.execute(
            """
            INSERT INTO records VALUES (
                'ev:same-node:same-block', 'legacy-doc', 'evidence',
                ?, NULL, 'Identical section title',
                '["Identical section title"]', 'same-block',
                'Legacy searchable text.', 'Legacy searchable text.',
                'paragraph', NULL, NULL, ?
            )
            """,
            (
                node_id,
                np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),
            ),
        )
        if documents is not None:
            connection.execute("""
                CREATE TABLE documents (
                    document_id TEXT PRIMARY KEY,
                    doc_name TEXT,
                    doc_title TEXT,
                    structure_path TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            connection.executemany(
                """
                INSERT INTO documents (
                    document_id, doc_name, doc_title, structure_path, record_count
                ) VALUES (?, ?, ?, ?, ?)
                """,
                documents,
            )


def _legacy_snapshot(db_path: Path) -> dict[str, list[tuple]]:
    with sqlite3.connect(db_path) as connection:
        table_names = [
            row[0]
            for row in connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type IN ('table', 'index', 'trigger', 'view')
                AND name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            )
        ]
        return {
            "schema": connection.execute(
                """
                SELECT type, name, tbl_name, sql FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """
            ).fetchall(),
            "metadata": connection.execute(
                "SELECT key, value FROM index_metadata ORDER BY key"
            ).fetchall(),
            "records": connection.execute(
                "SELECT * FROM records ORDER BY record_id"
            ).fetchall(),
            "documents": (
                connection.execute(
                    "SELECT * FROM documents ORDER BY document_id"
                ).fetchall()
                if "documents" in table_names
                else []
            ),
        }


class MultiDocumentRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "index.sqlite3"
        self.model_patcher = patch(
            "v2.retrieval.SentenceTransformer",
            FakeSentenceTransformer,
        )
        self.model_patcher.start()
        self.addCleanup(self.model_patcher.stop)

    def tearDown(self):
        self.temp_dir.cleanup()

    def index(self, tree: dict, filename: str) -> None:
        structure_path = _write_tree(self.root, tree, filename)
        build_index(structure_path, self.db_path)

    def test_indexing_two_documents_preserves_both_records_and_registry(self):
        self.index(_tree("doc-A", "Identical paragraph."), "a.json")
        self.index(_tree("doc-B", "Identical paragraph."), "b.json")

        with sqlite3.connect(self.db_path) as connection:
            records = connection.execute(
                "SELECT document_id, COUNT(*) FROM records GROUP BY document_id"
            ).fetchall()
            documents = connection.execute(
                "SELECT document_id, record_count FROM documents ORDER BY document_id"
            ).fetchall()

        self.assertEqual(records, [("doc-A", 2), ("doc-B", 2)])
        self.assertEqual(documents, [("doc-A", 2), ("doc-B", 2)])

    def test_reindex_replaces_only_its_document(self):
        self.index(_tree("doc-A", "Old A text."), "a-old.json")
        self.index(_tree("doc-B", "Unchanged B text."), "b.json")
        with sqlite3.connect(self.db_path) as connection:
            before_b = connection.execute(
                """
                SELECT record_id, source_text, embedding
                FROM records WHERE document_id = 'doc-B' ORDER BY record_id
                """
            ).fetchall()

        self.index(
            _tree("doc-A", "Updated A text.", summary="Updated summary"),
            "a-new.json",
        )

        with sqlite3.connect(self.db_path) as connection:
            after_b = connection.execute(
                """
                SELECT record_id, source_text, embedding
                FROM records WHERE document_id = 'doc-B' ORDER BY record_id
                """
            ).fetchall()
            a_texts = connection.execute(
                "SELECT source_text FROM records WHERE document_id = 'doc-A'"
            ).fetchall()
            documents = connection.execute(
                "SELECT document_id, record_count FROM documents ORDER BY document_id"
            ).fetchall()

        self.assertEqual(after_b, before_b)
        self.assertIn(("Updated A text.",), a_texts)
        self.assertNotIn(("Old A text.",), a_texts)
        self.assertEqual(documents, [("doc-A", 2), ("doc-B", 2)])

    def test_search_returns_owned_evidence_navigation_and_document_registry(self):
        self.index(
            _tree("doc-A", "The shared phrase appears in both files."),
            "a.json",
        )
        self.index(
            _tree("doc-B", "The shared phrase appears in both files."),
            "b.json",
        )

        result = search_index(
            "shared phrase",
            self.db_path,
            top_k=10,
            candidate_k=20,
        )

        self.assertEqual(
            {item["document_id"] for item in result["evidence"]},
            {"doc-A", "doc-B"},
        )
        self.assertEqual(
            {item["document_id"] for item in result["navigation_hits"]},
            {"doc-A", "doc-B"},
        )
        self.assertTrue(all(item["record_id"] for item in result["evidence"]))
        self.assertTrue(
            all(item["content_block_id"] == "same-block" for item in result["evidence"])
        )
        self.assertTrue(
            all(item["record_type"] == "navigation" for item in result["navigation_hits"])
        )
        self.assertTrue(
            all(item["record_type"] == "evidence" for item in result["evidence"])
        )
        self.assertEqual(
            {document["document_id"] for document in result["documents"]},
            {"doc-A", "doc-B"},
        )
        self.assertTrue(all(document["structure_path"] for document in result["documents"]))
        for item in result["evidence"] + result["navigation_hits"]:
            self.assertIn("doc_name", item)
            self.assertIn("doc_title", item)
            self.assertIn("section_path", item)
            self.assertIn("source_text", item)
            self.assertIn("fusion_score", item)
            self.assertIn("dense_score", item)
            self.assertIn("bm25_score", item)

    def test_navigation_records_can_fill_small_candidate_pool(self):
        tree = {
            "document_id": "doc-A",
            "doc_name": "doc-A",
            "doc_title": "Candidate competition",
            "preamble": [],
            "structure": [
                {
                    "node_id": f"navigation-{index}",
                    "title": f"Navigation {index}",
                    "summary": f"Navigation summary {index}",
                    "key_items": [],
                    "content_blocks": [],
                    "children": [],
                }
                for index in range(4)
            ] + [
                {
                    "node_id": "evidence-node",
                    "title": "Evidence section",
                    "summary": "Evidence section summary",
                    "key_items": [],
                    "content_blocks": [
                        {
                            "block_id": "target-block",
                            "type": "paragraph",
                            "text": "Relevant original source evidence.",
                        }
                    ],
                    "children": [],
                }
            ],
        }
        self.index(tree, "candidate-competition.json")
        evidence_record = next(
            record
            for record in make_records(tree)
            if record["record_type"] == "evidence"
        )

        navigation_embedding = np.asarray(
            [1.0, 0.0, 0.0],
            dtype=np.float32,
        ).tobytes()
        evidence_embedding = np.asarray(
            [0.0, 1.0, 0.0],
            dtype=np.float32,
        ).tobytes()
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE records
                SET searchable_text = 'needle',
                    embedding = ?
                WHERE record_type = 'navigation'
                """,
                (navigation_embedding,),
            )
            connection.execute(
                """
                UPDATE records
                SET searchable_text = 'evidence-only-token',
                    embedding = ?
                WHERE record_type = 'evidence'
                """,
                (evidence_embedding,),
            )

        five_candidates = search_index(
            "needle",
            self.db_path,
            top_k=10,
            candidate_k=5,
        )
        six_candidates = search_index(
            "needle",
            self.db_path,
            top_k=10,
            candidate_k=6,
        )

        self.assertTrue(five_candidates["navigation_hits"])
        self.assertTrue(
            all(
                item["record_type"] == "navigation"
                for item in five_candidates["navigation_hits"]
            )
        )
        self.assertEqual(five_candidates["evidence"], [])

        self.assertTrue(six_candidates["navigation_hits"])
        self.assertTrue(
            all(
                item["record_type"] == "navigation"
                for item in six_candidates["navigation_hits"]
            )
        )
        self.assertIn(
            evidence_record["record_id"],
            {
                item["record_id"]
                for item in six_candidates["evidence"]
            },
        )
        self.assertTrue(
            all(
                item["record_type"] == "evidence"
                for item in six_candidates["evidence"]
            )
        )

    def test_subtree_evidence_lookup_returns_original_ordered_records(self):
        tree = _tree("doc-A", "Text from later block.")
        tree["preamble"] = [
            {"block_id": "preamble-block", "type": "paragraph", "text": "Preamble."}
        ]
        tree["structure"][0]["node_id"] = "node-z"
        tree["structure"][0]["content_blocks"][0].update({
            "start_line": 5,
            "end_line": 6,
        })
        tree["structure"][0]["content_blocks"].extend([
            {
                "block_id": "tie-a",
                "type": "paragraph",
                "text": "Text from first block.",
                "start_line": 1,
                "end_line": 2,
            },
            {
                "block_id": "tie-b",
                "type": "paragraph",
                "text": "Text from middle block.",
                "start_line": 3,
                "end_line": 4,
            },
        ])
        tree["structure"].append({
            "node_id": "node-a",
            "title": "Earlier node",
            "summary": "Navigation summary.",
            "key_items": [],
            "content_blocks": [{
                "block_id": "block-a",
                "type": "paragraph",
                "text": "Text from second requested node.",
                "start_line": 1,
                "end_line": 2,
            }],
            "children": [],
        })
        self.index(tree, "a.json")
        self.index(
            _tree(
                "doc-B",
                "Text from another document.",
                node_id="node-z",
                block_id="tie-a",
            ),
            "b.json",
        )

        with sqlite3.connect(self.db_path) as connection:
            expected_ids = dict(connection.execute(
                """
                SELECT content_block_id, record_id
                FROM records
                WHERE document_id = 'doc-A'
                    AND record_type = 'evidence'
                    AND node_id != '__preamble__'
                """
            ).fetchall())

        results = get_subtree_evidence_records(
            self.db_path,
            "doc-A",
            [
                "node-z",
                "node-z",
                "node-a",
                "unknown-node",
                "__preamble__",
            ],
        )

        self.assertEqual(
            [item["node_id"] for item in results],
            ["node-z", "node-z", "node-z", "node-a"],
        )
        self.assertEqual(
            [item["content_block_id"] for item in results],
            ["tie-a", "tie-b", "same-block", "block-a"],
        )
        self.assertEqual(
            [item["record_id"] for item in results],
            [expected_ids[item["content_block_id"]] for item in results],
        )
        self.assertEqual(
            len({item["record_id"] for item in results}),
            len(results),
        )
        self.assertTrue(all(item["document_id"] == "doc-A" for item in results))
        self.assertTrue(all(item["record_type"] == "evidence" for item in results))
        self.assertEqual(
            [item["source_text"] for item in results],
            [
                "Text from first block.",
                "Text from middle block.",
                "Text from later block.",
                "Text from second requested node.",
            ],
        )
        self.assertTrue(
            all(item["text"] == item["source_text"] for item in results)
        )
        expected_fields = {
            "record_id",
            "document_id",
            "doc_name",
            "doc_title",
            "record_type",
            "node_id",
            "node_title",
            "section_path",
            "content_block_id",
            "text",
            "source_text",
            "block_type",
            "start_line",
            "end_line",
            "fusion_score",
            "dense_score",
            "bm25_score",
        }
        for item in results:
            self.assertEqual(set(item), expected_fields)
            self.assertIsNone(item["fusion_score"])
            self.assertIsNone(item["dense_score"])
            self.assertIsNone(item["bm25_score"])

        self.assertEqual(
            get_subtree_evidence_records(
                self.db_path,
                "doc-A",
                ["unknown-node"],
            ),
            [],
        )

    def test_subtree_evidence_lookup_empty_nodes_does_not_open_database(self):
        with patch("v2.retrieval.open_db") as open_database:
            results = get_subtree_evidence_records(
                self.db_path,
                "doc-A",
                [],
            )

        self.assertEqual(results, [])
        open_database.assert_not_called()

    def test_composite_ids_distinguish_identical_records_across_documents(self):
        tree_a = _tree("doc-A", "Same paragraph.")
        tree_b = _tree("doc-B", "Same paragraph.")
        records_a = make_records(tree_a)
        records_a_repeat = make_records(tree_a)
        records_b = make_records(tree_b)
        ids_a = {record["record_id"] for record in records_a}
        ids_b = {record["record_id"] for record in records_b}

        self.assertEqual(ids_a, {record["record_id"] for record in records_a_repeat})
        self.assertTrue(ids_a.isdisjoint(ids_b))
        self.assertEqual(
            records_a[1]["source_text"],
            records_b[1]["source_text"],
        )

    def test_failed_reindex_rolls_back_current_document_only(self):
        self.index(_tree("doc-A", "Original A text."), "a.json")
        self.index(_tree("doc-B", "Original B text."), "b.json")
        with sqlite3.connect(self.db_path) as connection:
            before_records = connection.execute(
                """
                SELECT record_id, document_id, source_text, embedding
                FROM records ORDER BY document_id, record_id
                """
            ).fetchall()
            before_documents = connection.execute(
                """
                SELECT document_id, doc_name, doc_title, structure_path,
                       record_count, indexed_at
                FROM documents ORDER BY document_id
                """
            ).fetchall()
            connection.execute("""
                CREATE TRIGGER reject_a_insert
                BEFORE INSERT ON records
                WHEN NEW.document_id = 'doc-A'
                BEGIN
                    SELECT RAISE(ABORT, 'simulated indexing failure');
                END
            """)

        changed_a = _tree("doc-A", "Replacement A text.")
        structure_path = _write_tree(self.root, changed_a, "a-updated.json")
        with self.assertRaises(sqlite3.IntegrityError):
            build_index(structure_path, self.db_path)

        with sqlite3.connect(self.db_path) as connection:
            after_records = connection.execute(
                """
                SELECT record_id, document_id, source_text, embedding
                FROM records ORDER BY document_id, record_id
                """
            ).fetchall()
            after_documents = connection.execute(
                """
                SELECT document_id, doc_name, doc_title, structure_path,
                       record_count, indexed_at
                FROM documents ORDER BY document_id
                """
            ).fetchall()

        self.assertEqual(after_records, before_records)
        self.assertEqual(after_documents, before_documents)

    def test_empty_document_reindex_removes_stale_records_and_stays_registered(self):
        self.index(_tree("doc-A", "Previously indexed."), "a-old.json")
        empty = {
            "document_id": "doc-A",
            "doc_name": "A",
            "doc_title": "Title A",
            "preamble": [],
            "structure": [],
        }
        self.index(empty, "a-empty.json")

        with sqlite3.connect(self.db_path) as connection:
            record_count = connection.execute(
                "SELECT COUNT(*) FROM records WHERE document_id = 'doc-A'"
            ).fetchone()[0]
            document = connection.execute(
                "SELECT document_id, record_count FROM documents"
            ).fetchone()

        self.assertEqual(record_count, 0)
        self.assertEqual(document, ("doc-A", 0))

    def test_schema_v1_migrates_existing_records_and_registers_structure_path(self):
        app_dir = self.root / "v2"
        database_path = app_dir / "storage" / "retrieval" / "index.sqlite3"
        structure_path = _write_tree(
            app_dir,
            _tree("legacy-doc", "Legacy searchable text."),
        )
        database_path.parent.mkdir(parents=True)
        with sqlite3.connect(database_path) as connection:
            connection.executescript("""
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
                );
                CREATE TABLE index_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)
            connection.executemany(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                [
                    ("schema_version", "1"),
                    ("document_id", "legacy-doc"),
                    ("doc_name", "legacy-doc"),
                    ("doc_title", "Title legacy-doc"),
                    ("record_count", "1"),
                    ("model_name", MODEL_NAME),
                    ("embedding_dimension", "3"),
                ],
            )
            connection.execute(
                """
                INSERT INTO records VALUES (
                    'ev:same-node:same-block', 'legacy-doc', 'evidence',
                    'same-node', NULL, 'Identical section title',
                    '["Identical section title"]', 'same-block',
                    'Legacy searchable text.', 'Legacy searchable text.',
                    'paragraph', NULL, NULL, ?
                )
                """,
                (np.asarray([1.0, 0.0, 0.0], dtype=np.float32).tobytes(),),
            )

        result = search_index(
            "legacy searchable text",
            database_path,
            top_k=5,
            candidate_k=10,
        )

        self.assertEqual(result["index_metadata"]["schema_version"], "2")
        self.assertEqual(result["documents"][0]["document_id"], "legacy-doc")
        self.assertEqual(
            result["documents"][0]["structure_path"],
            str(structure_path.resolve()),
        )
        self.assertEqual(result["evidence"][0]["document_id"], "legacy-doc")
        self.assertTrue(result["evidence"][0]["record_id"].startswith("record:"))
        with sqlite3.connect(database_path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT COUNT(*) FROM records WHERE document_id = 'legacy-doc'"
                ).fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                1,
            )

    def test_search_rejects_non_positive_top_k_and_candidate_k(self):
        cases = (
            ({"top_k": 0}, "top_k must be a positive integer"),
            ({"top_k": -1}, "top_k must be a positive integer"),
            ({"candidate_k": 0}, "candidate_k must be a positive integer"),
            ({"candidate_k": -1}, "candidate_k must be a positive integer"),
        )
        for kwargs, message in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    search_index("query", self.db_path, **kwargs)

    def test_search_rejects_corrupt_embedding_byte_length_with_record_identity(self):
        self.index(_tree("doc-A", "Corrupt length test."), "a.json")
        with sqlite3.connect(self.db_path) as connection:
            record_id = connection.execute(
                "SELECT record_id FROM records WHERE record_type = 'evidence'"
            ).fetchone()[0]
            connection.execute(
                "UPDATE records SET embedding = ? WHERE record_id = ?",
                (b"bad", record_id),
            )

        with self.assertRaises(ValueError) as raised:
            search_index("query", self.db_path)
        self.assertIn(f"record {record_id!r}", str(raised.exception))
        self.assertIn("invalid byte length", str(raised.exception))

    def test_search_rejects_non_finite_embedding_with_record_identity(self):
        self.index(_tree("doc-A", "Non-finite test."), "a.json")
        with sqlite3.connect(self.db_path) as connection:
            record_id = connection.execute(
                "SELECT record_id FROM records WHERE record_type = 'evidence'"
            ).fetchone()[0]
            bad_vector = np.asarray([np.nan, 0.0, 0.0], dtype=np.float32)
            connection.execute(
                "UPDATE records SET embedding = ? WHERE record_id = ?",
                (bad_vector.tobytes(), record_id),
            )

        with self.assertRaises(ValueError) as raised:
            search_index("query", self.db_path)
        self.assertIn(f"record {record_id!r}", str(raised.exception))
        self.assertIn("non-finite values", str(raised.exception))

    def test_empty_token_bm25_input_does_not_add_tied_candidates(self):
        self.index(_tree("doc-A", "Some dense evidence."), "a.json")
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("UPDATE records SET searchable_text = '!? --'")

        with (
            patch(
                "v2.retrieval.BM25Okapi",
                side_effect=AssertionError("BM25 must skip an empty token corpus"),
            ),
            patch(
                "v2.retrieval.reciprocal_rank_fusion",
                wraps=reciprocal_rank_fusion,
            ) as fusion,
        ):
            result = search_index("query", self.db_path, top_k=10, candidate_k=10)

        self.assertEqual(fusion.call_args.args[0][1], [])
        self.assertTrue(result["evidence"])
        self.assertTrue(all(item["bm25_score"] == 0 for item in result["evidence"]))

    def test_v1_migration_rejects_extra_registry_entries_without_changes(self):
        legacy_path = self.root / "legacy-extra.sqlite3"
        _create_v1_database(
            legacy_path,
            documents=[
                ("legacy-doc", "Legacy", "Legacy", "", 1),
                ("unindexed-doc", "Other", "Other", "/missing/structure.json", 7),
            ],
        )
        before = _legacy_snapshot(legacy_path)

        with self.assertRaisesRegex(
            ValueError,
            "legacy documents registry contains additional document entries",
        ):
            search_index("query", legacy_path)

        self.assertEqual(_legacy_snapshot(legacy_path), before)

    def test_failed_v1_migration_rolls_back_tables_metadata_and_records(self):
        legacy_path = self.root / "legacy-invalid.sqlite3"
        _create_v1_database(legacy_path, node_id="")
        before = _legacy_snapshot(legacy_path)

        with self.assertRaisesRegex(
            ValueError,
            "record is missing valid ownership or node/block identity",
        ):
            search_index("query", legacy_path)

        self.assertEqual(_legacy_snapshot(legacy_path), before)

    def test_records_document_id_index_exists(self):
        self.index(_tree("doc-A", "Index lookup test."), "a.json")
        with sqlite3.connect(self.db_path) as connection:
            index_columns = connection.execute(
                "PRAGMA index_info(idx_records_document_id)"
            ).fetchall()

        self.assertEqual([row[2] for row in index_columns], ["document_id"])

    def test_unknown_schema_version_is_not_interpreted_as_current(self):
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("""
                CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)
            """)
            connection.execute(
                "INSERT INTO index_metadata VALUES ('schema_version', '99')"
            )

        with self.assertRaisesRegex(ValueError, "Unsupported retrieval schema version"):
            search_index("query", self.db_path)

        with sqlite3.connect(self.db_path) as connection:
            self.assertEqual(
                connection.execute(
                    "SELECT value FROM index_metadata WHERE key = 'schema_version'"
                ).fetchone()[0],
                "99",
            )

    def test_fresh_database_has_v2_schema_and_corpus_metadata(self):
        self.index(_tree("doc-A", "Fresh index record."), "fresh.json")

        with sqlite3.connect(self.db_path) as connection:
            table_names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            metadata = dict(connection.execute("SELECT key, value FROM index_metadata"))
            record_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(records)")
            }
            document_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(documents)")
            }

        self.assertEqual(SCHEMA_VERSION, 2)
        self.assertTrue({"records", "documents", "index_metadata"} <= table_names)
        self.assertEqual(metadata["schema_version"], "2")
        self.assertEqual(metadata["model_name"], MODEL_NAME)
        self.assertEqual(metadata["embedding_dimension"], "3")
        self.assertTrue(
            {
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
            }
            <= record_columns
        )
        self.assertTrue(
            {
                "document_id",
                "doc_name",
                "doc_title",
                "structure_path",
                "record_count",
                "indexed_at",
            }
            <= document_columns
        )


if __name__ == "__main__":
    unittest.main()
