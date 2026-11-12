from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from . import client as api
from . import retrieval
from .document_queue import DocumentWorkQueue
from .query_routing import InvalidDocumentSelection, route_query
from .pipeline import PipelineStageError
from .evidence_evaluator import EvaluationValidationError


class FakeSentenceTransformer:
    dimension = 3

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name

    def get_sentence_embedding_dimension(self):
        return self.dimension

    def encode(self, texts, **kwargs):
        return np.tile(
            np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            (len(texts), 1),
        )


def _registry(*document_ids):
    return [
        {
            "document_id": document_id,
            "doc_name": document_id,
            "doc_title": f"Title {document_id}",
            "structure_path": f"/tmp/{document_id}.json",
            "record_count": 1,
            "indexed_at": "test",
        }
        for document_id in document_ids
    ]


class QueryRoutingTests(unittest.TestCase):
    def test_automatic_routing_uses_existing_hybrid_hits(self):
        registry = _registry("doc-A", "doc-B", "doc-C")
        retrieval_result = {
            "evidence": [
                {"document_id": "doc-A", "fusion_score": 0.81},
                {"document_id": "doc-B", "fusion_score": 0.54},
            ],
            "navigation_hits": [
                {"document_id": "doc-B", "fusion_score": 0.62},
            ],
        }
        with (
            patch("v2.query_routing.list_indexed_documents", return_value=registry),
            patch("v2.query_routing.search_index", return_value=retrieval_result) as search,
        ):
            routed = route_query("question", Path("index.sqlite3"))

        self.assertEqual(routed["mode"], "automatic")
        self.assertEqual(routed["selected_document_ids"], ["doc-A", "doc-B"])
        self.assertEqual(routed["document_scores"], {"doc-A": 0.81, "doc-B": 0.62})
        search.assert_called_once()

    def test_explicit_selection_validates_registry_without_hybrid_routing(self):
        with (
            patch("v2.query_routing.list_indexed_documents", return_value=_registry("doc-A", "doc-B")),
            patch("v2.query_routing.search_index") as search,
        ):
            routed = route_query(
                "question",
                Path("index.sqlite3"),
                document_ids=["doc-B", "doc-A"],
            )
        self.assertEqual(routed["mode"], "explicit")
        self.assertEqual(routed["selected_document_ids"], ["doc-B", "doc-A"])
        search.assert_not_called()

        with patch(
            "v2.query_routing.list_indexed_documents",
            return_value=_registry("doc-A"),
        ):
            with self.assertRaises(InvalidDocumentSelection):
                route_query("question", "index.sqlite3", document_ids=["missing"])

    def test_search_index_can_scope_hybrid_results_to_selected_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "index.sqlite3"
            structures = []
            for document_id, text in (("doc-A", "alpha evidence"), ("doc-B", "beta evidence")):
                tree = {
                    "document_id": document_id,
                    "doc_name": document_id,
                    "doc_title": document_id,
                    "preamble": [],
                    "structure": [{
                        "node_id": "same-node",
                        "title": "Section",
                        "summary": "Navigation summary",
                        "key_items": [],
                        "content_blocks": [{
                            "block_id": "same-block",
                            "type": "paragraph",
                            "text": text,
                        }],
                        "children": [],
                    }],
                }
                path = root / f"{document_id}.json"
                path.write_text(json.dumps(tree), encoding="utf-8")
                structures.append((path, document_id))

            with patch("v2.retrieval.SentenceTransformer", FakeSentenceTransformer):
                for path, _document_id in structures:
                    retrieval.build_index(path, db_path)
                result = retrieval.search_index(
                    "evidence",
                    db_path,
                    document_ids=["doc-B"],
                )

        self.assertEqual({item["document_id"] for item in result["evidence"]}, {"doc-B"})
        self.assertEqual({item["document_id"] for item in result["navigation_hits"]}, {"doc-B"})
        self.assertEqual([item["document_id"] for item in result["documents"]], ["doc-B"])


class DocumentQueueTests(unittest.TestCase):
    def test_same_document_jobs_are_sequential_in_submission_order(self):
        queue = DocumentWorkQueue()
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        order = []
        active = 0
        maximum_active = 0
        guard = threading.Lock()

        def operation(entry):
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            if entry.query_id == "query-1":
                first_started.set()
                release_first.wait(timeout=2)
            else:
                second_started.set()
            order.append(entry.query_id)
            with guard:
                active -= 1
            return entry.document_id

        first = queue.submit_query("query-1", "one", ["doc-A"], operation)["doc-A"]
        self.assertTrue(first_started.wait(timeout=2))
        second = queue.submit_query("query-2", "two", ["doc-A"], operation)["doc-A"]
        time.sleep(0.02)
        self.assertFalse(second_started.is_set())
        release_first.set()
        self.assertEqual(first.result(timeout=2), "doc-A")
        self.assertEqual(second.result(timeout=2), "doc-A")
        self.assertEqual(order, ["query-1", "query-2"])
        self.assertEqual(maximum_active, 1)
        self.assertEqual(queue.snapshot("query-2")["documents"]["doc-A"]["status"], "completed")

    def test_different_document_jobs_run_independently(self):
        queue = DocumentWorkQueue()
        both_started = threading.Barrier(2)

        def operation(entry):
            both_started.wait(timeout=2)
            return entry.document_id

        first = queue.submit_query("query-A", "one", ["doc-A"], operation)["doc-A"]
        second = queue.submit_query("query-B", "two", ["doc-B"], operation)["doc-B"]
        self.assertEqual({first.result(timeout=3), second.result(timeout=3)}, {"doc-A", "doc-B"})


class RoutedQueryApiTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(api.router)
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def test_multi_document_failure_isolated_and_provenance_is_aggregated(self):
        registry = _registry("doc-A", "doc-B")
        routed = {
            "mode": "explicit",
            "selected_document_ids": ["doc-A", "doc-B"],
            "document_scores": {},
            "registry": registry,
            "routing_retrieval": None,
        }

        def run_pipeline(query, **kwargs):
            document_id = kwargs["document_ids"][0]
            if document_id == "doc-A":
                raise PipelineStageError("evaluation", RuntimeError("provider error"), {
                    "retrieval": {"evidence": [{"document_id": "doc-A", "record_id": "A1"}],
                                  "navigation_hits": []},
                    "expanded_contexts": [{"document_id": "doc-A", "node_id": "node-A"}],
                    "loaded_documents": ["doc-A"],
                })
            return {
                "query": query,
                "retrieval": {
                    "evidence": [{"document_id": "doc-B", "record_id": "B1"}],
                    "navigation_hits": [],
                    "index_metadata": {"schema_version": "2"},
                },
                "expanded_contexts": [{"document_id": "doc-B", "node_id": "node-B"}],
                "evaluation": {"sufficiency": {"status": "sufficient"}},
                "loaded_documents": ["doc-B"],
            }

        with (
            patch("v2.client.route_query", return_value=routed),
            patch("v2.pipeline.run_pipeline", side_effect=run_pipeline) as pipeline,
        ):
            response = self.client.post(
                "/query",
                json={"query": "question", "document_ids": ["doc-A", "doc-B"]},
            )

        result = response.json()
        self.assertEqual(response.status_code, 207)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["routing"]["selected_document_ids"], ["doc-A", "doc-B"])
        self.assertEqual(result["retrieval"]["evidence"], [
            {"document_id": "doc-B", "record_id": "B1"},
            {"document_id": "doc-A", "record_id": "A1"},
        ])
        self.assertEqual(result["expanded_contexts"], [
            {"document_id": "doc-B", "node_id": "node-B"},
            {"document_id": "doc-A", "node_id": "node-A"},
        ])
        self.assertEqual(result["document_results"][0]["error"]["stage"], "evaluation")
        self.assertEqual(result["evaluation"]["by_document"], {
            "doc-B": {"sufficiency": {"status": "sufficient"}},
        })
        self.assertEqual(
            {item["document_id"]: item["status"] for item in result["document_results"]},
            {"doc-A": "failed", "doc-B": "completed"},
        )
        self.assertEqual(pipeline.call_count, 2)
        self.assertTrue(all(call.kwargs["document_ids"] in (["doc-A"], ["doc-B"]) for call in pipeline.call_args_list))

    def test_two_document_query_preserves_each_document_evaluation(self):
        registry = _registry("doc-A", "doc-B")
        routed = {
            "mode": "explicit", "selected_document_ids": ["doc-A", "doc-B"],
            "document_scores": {}, "registry": registry, "routing_retrieval": None,
        }
        def run_pipeline(query, **kwargs):
            document_id = kwargs["document_ids"][0]
            return {
                "query": query,
                "retrieval": {"evidence": [{"document_id": document_id, "record_id": document_id}],
                              "navigation_hits": []},
                "expanded_contexts": [{"document_id": document_id, "node_id": f"node-{document_id}"}],
                "evaluation": {"relevance": [{"document_id": document_id}]},
                "loaded_documents": [document_id],
            }
        with (
            patch("v2.client.route_query", return_value=routed),
            patch("v2.pipeline.run_pipeline", side_effect=run_pipeline),
        ):
            response = self.client.post("/query", json={"query": "compare", "document_ids": ["doc-A", "doc-B"]})
        result = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual({item["document_id"] for item in result["retrieval"]["evidence"]}, {"doc-A", "doc-B"})
        self.assertEqual(set(result["evaluation"]["by_document"]), {"doc-A", "doc-B"})

    def test_evaluator_diagnostics_are_serialized_without_research_content(self):
        registry = _registry("doc-A")
        routed = {
            "mode": "explicit", "selected_document_ids": ["doc-A"],
            "document_scores": {}, "registry": registry, "routing_retrieval": None,
        }
        diagnostic_payload = {
            "attempts": [{
                "attempt": 1,
                "expected_classification_count": 1,
                "actual_relevance_item_count": 0,
                "unique_classification_count": 0,
                "missing_evidence_ids": [["doc-A", "node-A", "block-A"]],
                "unexpected_evidence_ids": [],
                "duplicate_evidence_ids": [],
                "llm_response": {"finish_reason": "length", "prompt_tokens": 100,
                                 "completion_tokens": 20, "total_tokens": 120,
                                 "raw_response": "RAW MODEL RESPONSE"},
                "validation_error": "LLM did not classify every evidence block.",
                "prompt": "PRIVATE PROMPT",
                "evidence_text": "PRIVATE EVIDENCE",
            }],
            "query": "PRIVATE QUERY",
        }
        failure = EvaluationValidationError(
            "Evidence evaluation failed after 2 attempt(s): LLM did not classify every evidence block.",
            diagnostic_payload,
        )
        def run_pipeline(*_args, **_kwargs):
            raise PipelineStageError("evaluation", failure, {
                "retrieval": {"evidence": [], "navigation_hits": []},
                "expanded_contexts": [], "loaded_documents": ["doc-A"],
            })
        with (
            patch("v2.client.route_query", return_value=routed),
            patch("v2.pipeline.run_pipeline", side_effect=run_pipeline),
        ):
            response = self.client.post("/query", json={"query": "private query", "document_ids": ["doc-A"]})
        result = response.json()
        error = result["document_results"][0]["error"]
        diagnostics_json = json.dumps(error["diagnostics"])
        self.assertEqual(response.status_code, 503)
        self.assertEqual(error["diagnostics"]["attempts"][0]["llm_response"]["total_tokens"], 120)
        self.assertEqual(error["diagnostics"]["attempts"][0]["missing_evidence_ids"],
                         [["doc-A", "node-A", "block-A"]])
        for forbidden in ("RAW MODEL RESPONSE", "PRIVATE PROMPT", "PRIVATE EVIDENCE", "PRIVATE QUERY"):
            self.assertNotIn(forbidden, diagnostics_json)

    def test_invalid_explicit_document_selection_returns_422(self):
        with patch(
            "v2.client.route_query",
            side_effect=InvalidDocumentSelection("unknown document_id"),
        ):
            response = self.client.post(
                "/query",
                json={"query": "question", "document_ids": ["missing"]},
            )
        self.assertEqual(response.status_code, 422)

    def test_document_timeout_is_reported_and_running_job_keeps_its_slot(self):
        registry = _registry("doc-timeout")
        routed = {
            "mode": "explicit",
            "selected_document_ids": ["doc-timeout"],
            "document_scores": {},
            "registry": registry,
            "routing_retrieval": None,
        }
        finished = threading.Event()

        def slow_pipeline(*_args, **_kwargs):
            time.sleep(0.05)
            finished.set()
            return {
                "retrieval": {"evidence": [], "navigation_hits": []},
                "expanded_contexts": [],
                "evaluation": None,
                "loaded_documents": ["doc-timeout"],
            }

        with (
            patch("v2.client.route_query", return_value=routed),
            patch("v2.pipeline.run_pipeline", side_effect=slow_pipeline),
            patch.object(api, "QUERY_DOCUMENT_TIMEOUT_SECONDS", 0.005),
        ):
            response = self.client.post(
                "/query",
                json={"query": "question", "document_ids": ["doc-timeout"]},
            )

        result = response.json()
        query_id = result["query_id"]
        self.assertEqual(response.status_code, 503)
        self.assertEqual(result["document_results"][0]["status"], "timed_out")
        self.assertEqual(
            result["queue"]["documents"]["doc-timeout"]["status"],
            "timed_out",
        )
        self.assertTrue(finished.wait(timeout=2))
        self.assertEqual(
            api.DOCUMENT_WORK_QUEUE.snapshot(query_id)["documents"]["doc-timeout"]["status"],
            "timed_out",
        )


if __name__ == "__main__":
    unittest.main()
