import json
import tempfile
import unittest
from pathlib import Path

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
        retrieved = {
            "query": "What is the evidence?",
            "retrieval_status": "found",
            "evidence": [
                {
                    "node_id": "node_1",
                    "content_block_id": "block_2",
                    "text": "Evidence.",
                }
            ],
            "navigation_hits": [],
        }
        llm_response = {
            "relevance": [
                {
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
                    "evidence": [{"node_id": "node_1", "block_id": "block_2"}],
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


if __name__ == "__main__":
    unittest.main()
