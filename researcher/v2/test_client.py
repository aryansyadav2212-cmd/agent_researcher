from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from . import client as api
from . import retrieval
from .document_identity import DocumentIdentityStore


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


class FakeDocument:
    def __init__(self, source: bytes):
        self.source = source.decode("ascii")

    def export_to_dict(self):
        return {}

    def export_to_markdown(self):
        return f"## {self.source}\n\nUnique content for {self.source}."


class ApplicationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.db_path = self.root / "retrieval" / "index.sqlite3"
        self.documents_path = self.root / "documents"
        self.app = FastAPI()
        self.app.include_router(api.router)
        self.client = TestClient(self.app)

        self.identity_patcher = patch.object(
            api,
            "IDENTITY_STORE",
            DocumentIdentityStore(self.documents_path),
        )
        self.db_patcher = patch.object(api, "RETRIEVAL_DB_PATH", self.db_path)
        self.model_patcher = patch.object(
            retrieval,
            "SentenceTransformer",
            FakeSentenceTransformer,
        )
        self.identity_patcher.start()
        self.db_patcher.start()
        self.model_patcher.start()

        async def validate_upload(upload):
            return await upload.read()

        def process_pdf(path):
            return FakeDocument(Path(path).read_bytes())

        def save_structured_document(document, path):
            Path(path).write_text(
                json.dumps({"source": document.source}),
                encoding="utf-8",
            )

        self.upload_patches = [
            patch.object(api, "validate_pdf_upload", validate_upload),
            patch.object(api, "process_pdf", process_pdf),
            patch.object(api, "save_structured_document", save_structured_document),
        ]
        for item in self.upload_patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.upload_patches):
            item.stop()
        self.model_patcher.stop()
        self.db_patcher.stop()
        self.identity_patcher.stop()
        self.client.close()
        self.temp_dir.cleanup()

    def _upload(self, content: bytes, filename: str):
        return self.client.post(
            "/documents/upload",
            files={"file": (filename, content, "application/pdf")},
        )

    def _records_by_document(self):
        with sqlite3.connect(self.db_path) as connection:
            records: dict[str, list[str]] = {}
            for document_id, record_id in connection.execute(
                "SELECT document_id, record_id FROM records "
                "ORDER BY document_id, record_id"
            ):
                records.setdefault(document_id, []).append(record_id)
            return records

    def test_upload_indexes_multiple_documents_and_reindex_is_isolated(self):
        upload_a = self._upload(b"document A", "a.pdf")
        upload_b = self._upload(b"document B", "b.pdf")

        self.assertEqual(upload_a.status_code, 200, upload_a.text)
        self.assertEqual(upload_b.status_code, 200, upload_b.text)
        document_a = upload_a.json()["document_id"]
        document_b = upload_b.json()["document_id"]
        self.assertNotEqual(document_a, document_b)

        before_reindex = self._records_by_document()
        self.assertEqual(set(before_reindex), {document_a, document_b})
        self.assertEqual(len(before_reindex[document_a]), 1)
        self.assertEqual(len(before_reindex[document_b]), 1)

        reingested_a = self._upload(b"document A", "a.pdf")
        self.assertEqual(reingested_a.status_code, 200, reingested_a.text)
        self.assertEqual(reingested_a.json()["document_id"], document_a)

        after_reindex = self._records_by_document()
        self.assertEqual(after_reindex, before_reindex)

    def test_index_failure_does_not_report_success_and_saved_source_can_retry(self):
        with patch.object(api, "_index_structure", side_effect=RuntimeError("index failed")):
            response = self._upload(b"retryable document", "retry.pdf")

        self.assertEqual(response.status_code, 500)
        self.assertIn("indexing failed", response.json()["detail"])
        self.assertEqual(list(self.documents_path.glob("*/structure.json")).__len__(), 1)
        self.assertFalse(self.db_path.exists())

        retry = self._upload(b"retryable document", "retry.pdf")
        self.assertEqual(retry.status_code, 200, retry.text)
        self.assertTrue(self.db_path.exists())

    def test_arxiv_import_also_indexes_the_processed_document(self):
        async def download(_url):
            return b"arxiv document"

        with (
            patch.object(api, "normalize_arxiv_url", return_value="https://arxiv.org/pdf/1234.5678"),
            patch.object(api, "download_arxiv_pdf", download),
        ):
            response = self.client.post(
                "/documents/arxiv",
                params={"url": "https://arxiv.org/abs/1234.5678"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        document_id = response.json()["document_id"]
        self.assertEqual(set(self._records_by_document()), {document_id})

    def test_query_api_returns_pipeline_provenance_and_uses_existing_pipeline(self):
        pipeline_result = {
            "query": "compare the methods",
            "retrieval": {
                "documents": [{"document_id": "doc-one"}],
                "evidence": [{"document_id": "doc-one", "record_id": "r-one"}],
            },
            "expanded_contexts": [{
                "document_id": "doc-one",
                "node_id": "n-one",
                "anchor_block_id": "b-one",
            }],
            "evaluation": {"sufficiency": {"status": "sufficient"}},
            "loaded_documents": ["doc-one"],
        }
        routed = {
            "mode": "explicit",
            "selected_document_ids": ["doc-one"],
            "document_scores": {},
            "registry": [{"document_id": "doc-one"}],
            "routing_retrieval": None,
        }
        with (
            patch("v2.client.route_query", return_value=routed),
            patch("v2.pipeline.run_pipeline", return_value=pipeline_result) as pipeline,
        ):
            response = self.client.post(
                "/query",
                json={"query": "compare the methods", "document_ids": ["doc-one"]},
            )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["retrieval"]["evidence"], pipeline_result["retrieval"]["evidence"])
        self.assertEqual(response.json()["expanded_contexts"], pipeline_result["expanded_contexts"])
        self.assertEqual(response.json()["evaluation"], pipeline_result["evaluation"])
        self.assertEqual(response.json()["routing"]["selected_document_ids"], ["doc-one"])
        pipeline.assert_called_once()
        args, kwargs = pipeline.call_args
        self.assertEqual(args[0], "compare the methods")
        self.assertTrue(callable(kwargs["llm_call"]))
        self.assertEqual(kwargs["db_path"], self.db_path)

    def test_query_api_rejects_blank_query_and_reports_missing_documents(self):
        self.assertEqual(
            self.client.post("/query", json={"query": "   "}).status_code,
            422,
        )
        self.assertEqual(self.client.post("/query", json={}).status_code, 422)

        with patch(
            "v2.client.route_query",
            return_value={
                "mode": "explicit",
                "selected_document_ids": ["doc-A"],
                "document_scores": {},
                "registry": [{"document_id": "doc-A"}],
                "routing_retrieval": None,
            },
        ), patch(
            "v2.pipeline.run_pipeline",
            side_effect=FileNotFoundError("structure unavailable"),
        ):
            response = self.client.post(
                "/query",
                json={"query": "find evidence", "document_ids": ["doc-A"]},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "failed")
        self.assertIn("structure unavailable", response.json()["document_results"][0]["error"]["message"])


if __name__ == "__main__":
    unittest.main()
