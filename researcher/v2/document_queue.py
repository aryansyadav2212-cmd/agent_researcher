"""In-process FIFO work queues with one worker per document identity."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import threading
import time
from typing import Any, Callable


@dataclass(frozen=True)
class DocumentQueueEntry:
    query_id: str
    query: str
    document_id: str
    submitted_at: str


class DocumentWorkQueue:
    """Serialize work per document and run unrelated documents concurrently.

    Queue state and executors are process-local. They are not durable and are
    not shared by multiple application workers.
    """

    def __init__(self, *, retained_queries: int = 256):
        self._guard = threading.RLock()
        self._executors: dict[str, ThreadPoolExecutor] = {}
        self._pending_counts: dict[str, int] = {}
        self._queries: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._retained_queries = retained_queries

    def submit_query(
        self,
        query_id: str,
        query: str,
        document_ids: list[str],
        operation: Callable[[DocumentQueueEntry], Any],
    ) -> dict[str, Future]:
        if not query_id or not query.strip():
            raise ValueError("query_id and query are required.")
        if not document_ids:
            raise ValueError("document_ids must be non-empty and unique.")
        if any(not isinstance(doc_id, str) or not doc_id.strip() for doc_id in document_ids):
            raise ValueError("Every document_id must be a non-empty string.")
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document_ids must be non-empty and unique.")

        submitted_at = datetime.now(timezone.utc).isoformat()
        entries = {
            document_id: DocumentQueueEntry(
                query_id=query_id,
                query=query,
                document_id=document_id,
                submitted_at=submitted_at,
            )
            for document_id in document_ids
        }

        with self._guard:
            if query_id in self._queries:
                raise ValueError(f"Duplicate query_id: {query_id}")
            self._queries[query_id] = {
                "query_id": query_id,
                "query": query,
                "submitted_at": submitted_at,
                "documents": {
                    doc_id: {"status": "queued", "error": None,
                             "stage": "queued", "diagnostics": {},
                             "_stage_started": time.monotonic()}
                    for doc_id in document_ids
                },
            }
            self._prune()

            futures: dict[str, Future] = {}
            for document_id in document_ids:
                executor = self._executors.get(document_id)
                if executor is None:
                    executor = ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix=f"document-{document_id[:8]}",
                    )
                    self._executors[document_id] = executor
                    self._pending_counts[document_id] = 0
                self._pending_counts[document_id] += 1
                futures[document_id] = executor.submit(
                    self._execute,
                    entries[document_id],
                    operation,
                )
                futures[document_id].add_done_callback(
                    lambda future, qid=query_id, did=document_id, pool=executor:
                    self._on_done(qid, did, pool, future)
                )
            return futures

    def _execute(
        self,
        entry: DocumentQueueEntry,
        operation: Callable[[DocumentQueueEntry], Any],
    ) -> Any:
        with self._guard:
            state = self._queries[entry.query_id]["documents"][entry.document_id]
            state["status"] = "running"
            state["stage"] = "running"
            state["_stage_started"] = time.monotonic()
        try:
            result = operation(entry)
        except BaseException as exc:
            with self._guard:
                state = self._queries.get(entry.query_id, {}).get("documents", {}).get(entry.document_id)
                if state is not None:
                    if state["status"] != "timed_out":
                        state["status"] = "failed"
                    state["error"] = f"{type(exc).__name__}: {exc}"
            raise
        with self._guard:
            state = self._queries.get(entry.query_id, {}).get("documents", {}).get(entry.document_id)
            if state is not None and state["status"] != "timed_out":
                state["status"] = "completed"
        return result

    def _on_done(
        self,
        query_id: str,
        document_id: str,
        executor: ThreadPoolExecutor,
        future: Future,
    ) -> None:
        with self._guard:
            state = self._queries.get(query_id, {}).get("documents", {}).get(document_id)
            if future.cancelled() and state is not None:
                state["status"] = "cancelled"
            if self._executors.get(document_id) is executor:
                pending = self._pending_counts[document_id] - 1
                if pending <= 0:
                    self._pending_counts.pop(document_id, None)
                    self._executors.pop(document_id, None)
                    executor.shutdown(wait=False)
                else:
                    self._pending_counts[document_id] = pending

    def mark_timed_out(self, query_id: str, document_id: str) -> None:
        with self._guard:
            state = self._queries.get(query_id, {}).get("documents", {}).get(document_id)
            if state is not None and state["status"] == "running":
                state["status"] = "timed_out"

    def update_progress(self, query_id: str, document_id: str, stage: str,
                        diagnostics: dict[str, Any]) -> None:
        with self._guard:
            state = self._queries.get(query_id, {}).get("documents", {}).get(document_id)
            if state is not None:
                state["stage"] = stage
                state["diagnostics"] = dict(diagnostics)
                state["_stage_started"] = time.monotonic()

    def snapshot(self, query_id: str) -> dict[str, Any] | None:
        with self._guard:
            state = self._queries.get(query_id)
            if state is None:
                return None
            documents = {}
            for document_id, document_state in state["documents"].items():
                public_state = {key: value for key, value in document_state.items()
                                if not key.startswith("_")}
                public_state["stage_elapsed_seconds"] = round(
                    max(0.0, time.monotonic() - document_state.get("_stage_started", time.monotonic())), 3
                )
                documents[document_id] = public_state
            return {
                **state,
                "documents": documents,
            }

    def _prune(self) -> None:
        while len(self._queries) > self._retained_queries:
            query_id, query = next(iter(self._queries.items()))
            if any(
                state["status"] in {"queued", "running"}
                for state in query["documents"].values()
            ):
                self._queries.move_to_end(query_id)
                if all(
                    any(
                        state["status"] in {"queued", "running"}
                        for state in item["documents"].values()
                    )
                    for item in self._queries.values()
                ):
                    break
            else:
                self._queries.popitem(last=False)
