import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import retrieval
from .context_expander import ContextExpander
from .pipeline import run_pipeline


class PipelineIntegrationTest(unittest.TestCase):
    def test_runs_retrieval_context_expansion_and_evaluation(self):
        document = {
            "document_id": "doc_test",
            "doc_name": "Test",
            "doc_title": "Test document",
            "preamble": [],
            "structure": [
                {
                    "node_id": "node_1",
                    "title": "Section",
                    "content_blocks": [
                        {"block_id": "block_1", "type": "paragraph", "text": "First."},
                        {"block_id": "block_2", "type": "paragraph", "text": "Evidence."},
                        {"block_id": "block_3", "type": "paragraph", "text": "Last."},
                    ],
                    "children": [],
                }
            ],
        }
        evidence_record = next(
            record
            for record in retrieval.make_records(document)
            if record["record_type"] == "evidence"
            and record["node_id"] == "node_1"
            and record["content_block_id"] == "block_2"
        )
        retrieved = {
            "query": "What is the evidence?",
            "retrieval_status": "found",
            "evidence": [
                {
                    "record_id": evidence_record["record_id"],
                    "document_id": evidence_record["document_id"],
                    "record_type": evidence_record["record_type"],
                    "node_id": evidence_record["node_id"],
                    "content_block_id": evidence_record["content_block_id"],
                    "text": evidence_record["source_text"],
                    "source_text": evidence_record["source_text"],
                }
            ],
            "navigation_hits": [],
        }
        llm_response = {
            "relevance": [
                {
                    "document_id": "doc_test",
                    "node_id": "node_1",
                    "block_id": block_id,
                    "classification": "relevant" if block_id == "block_2" else "supporting",
                    "reason": "Supports the query.",
                }
                for block_id in ("block_1", "block_2", "block_3")
            ],
            "coverage": [
                {
                    "requirement": "Identify the evidence",
                    "status": "supported",
                    "evidence": [{
                        "document_id": "doc_test",
                        "node_id": "node_1",
                        "block_id": "block_2",
                    }],
                }
            ],
            "sufficiency": {
                "status": "sufficient",
                "reason": "The evidence is present.",
            },
        }
        calls = []

        def fake_retriever(query, db_path, **kwargs):
            calls.append((query, db_path, kwargs))
            return retrieved

        with tempfile.TemporaryDirectory() as temp_dir:
            structure_path = Path(temp_dir) / "structure.json"
            structure_path.write_text(json.dumps(document), encoding="utf-8")

            result = run_pipeline(
                "What is the evidence?",
                structure_path=structure_path,
                db_path=Path(temp_dir) / "index.sqlite3",
                retriever=fake_retriever,
                llm_call=lambda prompt: json.dumps(llm_response),
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "What is the evidence?")
        self.assertEqual(
            [block["block_id"] for block in result["expanded_contexts"][0]["blocks"]],
            ["block_1", "block_2", "block_3"],
        )
        self.assertEqual(result["evaluation"]["sufficiency"]["status"], "sufficient")
        self.assertEqual(result["retrieval"], retrieved)


class NavigationSubtreePipelineTests(unittest.TestCase):
    def _node(
        self,
        node_id,
        block_ids,
        *,
        children=None,
        title=None,
    ):
        return {
            "node_id": node_id,
            "title": title or node_id,
            "summary": f"Summary for {node_id}",
            "key_items": [],
            "content_blocks": [
                {
                    "block_id": block_id,
                    "type": "paragraph",
                    "text": f"Text for {block_id}",
                    "start_line": index * 2 + 1,
                    "end_line": index * 2 + 1,
                }
                for index, block_id in enumerate(block_ids)
            ],
            "children": children or [],
        }

    def _document(self, document_id, nodes):
        return {
            "document_id": document_id,
            "doc_name": document_id,
            "doc_title": f"Title {document_id}",
            "preamble": [],
            "structure": nodes,
        }

    def _write_index_and_documents(self, root, documents):
        db_path = root / "index.sqlite3"
        document_registry = []
        records_by_id = {}

        with sqlite3.connect(db_path) as connection:
            retrieval._create_current_tables(connection)
            connection.execute(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                ("schema_version", str(retrieval.SCHEMA_VERSION)),
            )

            for document in documents:
                structure_path = root / f"{document['document_id']}.json"
                structure_path.write_text(
                    json.dumps(document),
                    encoding="utf-8",
                )
                document_records = retrieval.make_records(document)
                records_by_id.update({
                    record["record_id"]: record
                    for record in document_records
                })
                document_registry.append({
                    "document_id": document["document_id"],
                    "doc_name": document["doc_name"],
                    "doc_title": document["doc_title"],
                    "structure_path": str(structure_path),
                    "record_count": len(document_records),
                    "indexed_at": "test",
                })

                connection.execute(
                    """
                    INSERT INTO documents (
                        document_id, doc_name, doc_title, structure_path,
                        record_count, indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        document["document_id"],
                        document["doc_name"],
                        document["doc_title"],
                        str(structure_path),
                        len(document_records),
                        "test",
                    ),
                )

                connection.executemany(
                    """
                    INSERT INTO records (
                        record_id, document_id, record_type, node_id,
                        parent_id, node_title, section_path,
                        content_block_id, searchable_text, source_text,
                        block_type, start_line, end_line, embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            record["record_id"],
                            record["document_id"],
                            record["record_type"],
                            record["node_id"],
                            record["parent_id"],
                            record["node_title"],
                            json.dumps(record["section_path"]),
                            record["content_block_id"],
                            record["text"],
                            record["source_text"],
                            record["block_type"],
                            record["start_line"],
                            record["end_line"],
                            b"test-embedding",
                        )
                        for record in document_records
                    ],
                )

        return db_path, document_registry, records_by_id

    def _evidence_hit(self, record, **scores):
        return {
            "record_id": record["record_id"],
            "document_id": record["document_id"],
            "doc_name": record["document_id"],
            "doc_title": f"Title {record['document_id']}",
            "record_type": "evidence",
            "node_id": record["node_id"],
            "node_title": record["node_title"],
            "section_path": record["section_path"],
            "content_block_id": record["content_block_id"],
            "text": record["source_text"],
            "source_text": record["source_text"],
            "block_type": record["block_type"],
            "start_line": record["start_line"],
            "end_line": record["end_line"],
            "fusion_score": scores.get("fusion_score", 0.5),
            "dense_score": scores.get("dense_score", 0.4),
            "bm25_score": scores.get("bm25_score", 0.3),
        }

    def _retrieval_result(self, documents, evidence=None, navigation=None):
        return {
            "evidence": evidence or [],
            "navigation_hits": navigation or [],
            "documents": documents,
        }

    def _llm_response(self, prompt):
        evidence_json = prompt.split("EVIDENCE BLOCKS:\n", 1)[1].split(
            "\n\nPerform THREE evaluations.",
            1,
        )[0]
        evidence = json.loads(evidence_json)
        return json.dumps({
            "relevance": [
                {
                    "document_id": item["document_id"],
                    "node_id": item["node_id"],
                    "block_id": item["block_id"],
                    "classification": "supporting",
                    "reason": "Test evidence.",
                }
                for item in evidence
            ],
            "coverage": [],
            "sufficiency": {
                "status": "sufficient",
                "reason": "Test evaluation.",
            },
        })

    def _run_pipeline(self, db_path, retrieval_result, llm_call=None):
        return run_pipeline(
            "test query",
            db_path=db_path,
            retriever=lambda *args, **kwargs: retrieval_result,
            llm_call=llm_call or self._llm_response,
        )

    def test_navigation_only_retrieval_expands_subtree_evidence(self):
        child = self._node("child", ["child-block"])
        root_node = self._node("root", ["root-block"], children=[child])
        document = self._document("doc-A", [root_node])

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path, registry, records = self._write_index_and_documents(
                Path(temp_dir),
                [document],
            )
            result = self._run_pipeline(
                db_path,
                self._retrieval_result(
                    registry,
                    navigation=[{"document_id": "doc-A", "node_id": "root"}],
                ),
            )

        expected_records = [
            record
            for record in records.values()
            if record["record_type"] == "evidence"
            and record["document_id"] == "doc-A"
            and record["node_id"] in {"root", "child"}
        ]
        node_order = {"root": 0, "child": 1}
        expected_records.sort(key=lambda record: node_order[record["node_id"]])
        self.assertEqual(
            [context["record_id"] for context in result["expanded_contexts"]],
            [record["record_id"] for record in expected_records],
        )
        self.assertEqual(result["evaluation"]["sufficiency"]["status"], "sufficient")
        self.assertEqual(result["loaded_documents"], ["doc-A"])

    def test_direct_evidence_wins_overlap_and_keeps_retrieval_scores(self):
        document = self._document("doc-A", [self._node("root", ["block"])])

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path, registry, records = self._write_index_and_documents(
                Path(temp_dir),
                [document],
            )
            record = next(
                item for item in records.values()
                if item["record_type"] == "evidence"
            )
            direct_hit = self._evidence_hit(
                record,
                fusion_score=0.91,
                dense_score=0.82,
                bm25_score=0.73,
            )
            duplicate_direct_hit = self._evidence_hit(
                record,
                fusion_score=0.61,
                dense_score=0.52,
                bm25_score=0.43,
            )
            seen_evidence = []
            real_expander = ContextExpander

            class RecordingExpander:
                def __init__(self, evidence, structure_index):
                    seen_evidence.append(evidence)
                    self.expander = real_expander(evidence, structure_index)

                def expand_context(self, **kwargs):
                    return self.expander.expand_context(**kwargs)

            with patch("v2.pipeline.ContextExpander", RecordingExpander):
                result = self._run_pipeline(
                    db_path,
                    self._retrieval_result(
                        registry,
                        evidence=[direct_hit, duplicate_direct_hit],
                        navigation=[{"document_id": "doc-A", "node_id": "root"}],
                    ),
                )

        self.assertEqual(
            [context["record_id"] for context in result["expanded_contexts"]],
            [record["record_id"]],
        )
        self.assertEqual(len(seen_evidence), 1)
        self.assertIs(seen_evidence[0], direct_hit)
        self.assertEqual(seen_evidence[0]["fusion_score"], 0.91)
        self.assertEqual(seen_evidence[0]["dense_score"], 0.82)
        self.assertEqual(seen_evidence[0]["bm25_score"], 0.73)

    def test_direct_evidence_outside_selected_subtree_is_preserved(self):
        document = self._document(
            "doc-A",
            [
                self._node("root-A", ["block-A1", "block-A2"]),
                self._node("root-B", ["block-B1"]),
            ],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path, registry, records = self._write_index_and_documents(
                Path(temp_dir),
                [document],
            )
            evidence_by_block = {
                record["content_block_id"]: record
                for record in records.values()
                if record["record_type"] == "evidence"
            }
            direct_a = self._evidence_hit(
                evidence_by_block["block-A1"],
                fusion_score=0.93,
                dense_score=0.84,
                bm25_score=0.75,
            )
            direct_b = self._evidence_hit(
                evidence_by_block["block-B1"],
                fusion_score=0.62,
                dense_score=0.53,
                bm25_score=0.44,
            )
            seen_evidence = []
            real_expander = ContextExpander

            class RecordingExpander:
                def __init__(self, evidence, structure_index):
                    seen_evidence.append(evidence)
                    self.expander = real_expander(evidence, structure_index)

                def expand_context(self, **kwargs):
                    return self.expander.expand_context(**kwargs)

            with patch("v2.pipeline.ContextExpander", RecordingExpander):
                result = self._run_pipeline(
                    db_path,
                    self._retrieval_result(
                        registry,
                        evidence=[direct_a, direct_b],
                        navigation=[{
                            "document_id": "doc-A",
                            "node_id": "root-A",
                        }],
                    ),
                )

        expected_ids = [
            evidence_by_block["block-A1"]["record_id"],
            evidence_by_block["block-B1"]["record_id"],
            evidence_by_block["block-A2"]["record_id"],
        ]
        actual_ids = [evidence["record_id"] for evidence in seen_evidence]
        self.assertEqual(actual_ids, expected_ids)
        self.assertEqual(len(set(actual_ids)), 3)
        self.assertEqual(
            [evidence["record_id"] for evidence in result["expanded_contexts"]],
            expected_ids,
        )

        self.assertIs(seen_evidence[0], direct_a)
        self.assertEqual(direct_a["fusion_score"], 0.93)
        self.assertEqual(direct_a["dense_score"], 0.84)
        self.assertEqual(direct_a["bm25_score"], 0.75)
        self.assertIs(seen_evidence[1], direct_b)
        self.assertEqual(direct_b["fusion_score"], 0.62)
        self.assertEqual(direct_b["dense_score"], 0.53)
        self.assertEqual(direct_b["bm25_score"], 0.44)

    def test_overlapping_navigation_subtrees_do_not_duplicate_evidence(self):
        child = self._node("child", ["child-block"])
        document = self._document(
            "doc-A",
            [self._node("root", ["root-block"], children=[child])],
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path, registry, records = self._write_index_and_documents(
                Path(temp_dir),
                [document],
            )
            with patch(
                "v2.retrieval.get_subtree_evidence_records",
                wraps=retrieval.get_subtree_evidence_records,
            ) as subtree_lookup:
                result = self._run_pipeline(
                    db_path,
                    self._retrieval_result(
                        registry,
                        navigation=[
                            {"document_id": "doc-A", "node_id": "root"},
                            {"document_id": "doc-A", "node_id": "child"},
                        ],
                    ),
                )

        expected_ids = {
            record["record_id"]
            for record in records.values()
            if record["record_type"] == "evidence"
        }
        actual_ids = [context["record_id"] for context in result["expanded_contexts"]]
        self.assertEqual(set(actual_ids), expected_ids)
        self.assertEqual(len(actual_ids), len(expected_ids))
        subtree_lookup.assert_called_once()
        self.assertEqual(
            subtree_lookup.call_args.args[2],
            ["root", "child"],
        )

    def test_identical_node_ids_remain_document_specific(self):
        documents = [
            self._document(
                "doc-A",
                [self._node("same-node", ["block-A"])],
            ),
            self._document(
                "doc-B",
                [self._node("same-node", ["block-B"])],
            ),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path, registry, records = self._write_index_and_documents(
                Path(temp_dir),
                documents,
            )
            result = self._run_pipeline(
                db_path,
                self._retrieval_result(
                    registry,
                    navigation=[
                        {"document_id": "doc-A", "node_id": "same-node"},
                        {"document_id": "doc-B", "node_id": "same-node"},
                    ],
                ),
            )

        expected = {
            (
                record["document_id"],
                record["record_id"],
            )
            for record in records.values()
            if record["record_type"] == "evidence"
        }
        actual = {
            (context["document_id"], context["record_id"])
            for context in result["expanded_contexts"]
        }
        self.assertEqual(actual, expected)
        self.assertEqual(len(result["expanded_contexts"]), 2)

    def test_empty_evidence_and_navigation_skips_evaluation_and_loading(self):
        llm_calls = []
        retrieval_result = self._retrieval_result(
            [],
            evidence=[],
            navigation=[],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self._run_pipeline(
                Path(temp_dir) / "missing.sqlite3",
                retrieval_result,
                llm_call=lambda prompt: llm_calls.append(prompt),
            )

        self.assertEqual(result["query"], "test query")
        self.assertIs(result["retrieval"], retrieval_result)
        self.assertEqual(result["expanded_contexts"], [])
        self.assertIsNone(result["evaluation"])
        self.assertEqual(result["loaded_documents"], [])
        self.assertEqual(llm_calls, [])


if __name__ == "__main__":
    unittest.main()
