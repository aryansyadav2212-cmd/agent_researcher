
"""
Main entry point for document indexing, retrieval,
and conversation flows.
"""

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
import logging
import uuid
from pathlib import Path
import json
import os

import httpx
from fastapi import Query, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from .document_identity import DocumentIdentity, DocumentIdentityStore
from .document_utils import (
    validate_pdf_upload,
    process_pdf,
    save_structured_document,
    normalize_arxiv_url,
    MAX_FILE_SIZE,
)
from .markdown_extractor import extract_markdown_structure
from .document_queue import DocumentQueueEntry, DocumentWorkQueue
from .evidence_evaluator import EvaluationValidationError, LLMCallResult
from .query_routing import InvalidDocumentSelection, route_query

router = APIRouter()

DOCUMENTS_DIR = Path("storage/documents")
IDENTITY_STORE = DocumentIdentityStore(DOCUMENTS_DIR)
RETRIEVAL_DB_PATH = Path(__file__).resolve().parent / "storage" / "retrieval" / "index.sqlite3"
QUERY_DOCUMENT_TIMEOUT_SECONDS = 180
DOCUMENT_WORK_QUEUE = DocumentWorkQueue()


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    document_ids: list[str] | None = Field(default=None, max_length=32)


class EvaluationProviderError(RuntimeError):
    """Raised when the configured evaluation service cannot be used."""

    def __init__(self, message: str, provider_error_type: str | None = None):
        super().__init__(message)
        self.provider_error_type = provider_error_type


def _index_structure(structure_path: Path, db_path: Path) -> None:
    """Index one persisted structure using the shared retrieval implementation."""
    from .retrieval import build_index

    build_index(structure_path, db_path)


def _openai_evaluator_call(prompt: str) -> LLMCallResult:
    """Call the configured OpenAI chat model used for evidence evaluation."""
    try:
        from openai import OpenAI

        client = OpenAI(timeout=45.0, max_retries=0)
        response = client.chat.completions.create(
            model=os.environ.get("RESEARCHER_EVALUATION_MODEL", "gpt-4.1-mini"),
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
    except Exception as exc:
        raise EvaluationProviderError(
            "Evidence evaluation service is unavailable.",
            provider_error_type=type(exc).__name__,
        ) from exc

    choice = response.choices[0]
    content = choice.message.content
    if not content:
        raise EvaluationProviderError("Evidence evaluation service returned an empty response.")
    usage = response.usage
    return LLMCallResult(
        content=content,
        finish_reason=choice.finish_reason,
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )


def _serialize_evaluation_diagnostics(value):
    """Whitelist evaluator diagnostics for safe, JSON-only API output."""
    if not isinstance(value, dict) or not isinstance(value.get("attempts"), list):
        return None

    attempts = []
    id_fields = (
        "missing_evidence_ids",
        "unexpected_evidence_ids",
        "duplicate_evidence_ids",
    )
    count_fields = (
        "expected_classification_count",
        "actual_relevance_item_count",
        "unique_classification_count",
    )
    for source in value["attempts"]:
        if not isinstance(source, dict):
            continue
        attempt = {"attempt": source.get("attempt")}
        for field in count_fields:
            number = source.get(field)
            attempt[field] = number if isinstance(number, int) and not isinstance(number, bool) else None
        for field in id_fields:
            triples = source.get(field, [])
            attempt[field] = [
                [part for part in triple]
                for triple in triples
                if isinstance(triple, (list, tuple))
                and len(triple) == 3
                and all(isinstance(part, str) for part in triple)
            ] if isinstance(triples, list) else []
        metadata = source.get("llm_response")
        if isinstance(metadata, dict):
            finish_reason = metadata.get("finish_reason")
            attempt["llm_response"] = {
                "finish_reason": finish_reason if isinstance(finish_reason, str) else None,
                **{
                    field: metadata.get(field)
                    if isinstance(metadata.get(field), int) and not isinstance(metadata.get(field), bool)
                    else None
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens")
                },
            }
        else:
            attempt["llm_response"] = None
        validation_error = source.get("validation_error")
        attempt["validation_error"] = (
            _safe_validation_message(validation_error)
            if isinstance(validation_error, str) else None
        )
        attempts.append(attempt)
    return {"attempts": attempts}


def _safe_validation_message(message: str) -> str:
    """Keep the validation category while IDs remain in their structured fields."""
    if message.startswith("LLM returned unknown evidence ID:"):
        return "LLM returned unknown evidence ID."
    if message.startswith("Duplicate relevance evaluation:"):
        return "Duplicate relevance evaluation."
    if message.startswith("Coverage references unknown evidence:"):
        return "Coverage references unknown evidence."
    return message


def _index_or_raise(structure_path: Path) -> None:
    try:
        _index_structure(structure_path, RETRIEVAL_DB_PATH)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Document processing succeeded, but retrieval indexing failed. "
                "The saved document can be retried by uploading the same PDF again."
            ),
        ) from exc


def _save_pageindex_structure(
    document,
    identity: DocumentIdentity,
    doc_name: str,
) -> Path:
    """Extract and persist the PageIndex-compatible tree for a processed PDF."""
    docling_data = document.export_to_dict()
    previous_structure = None
    if identity.structure_path.exists():
        with identity.structure_path.open("r", encoding="utf-8") as file:
            previous = json.load(file)
        if isinstance(previous, dict) and isinstance(previous.get("structure"), list):
            previous_structure = previous

    tree = extract_markdown_structure(
        document.export_to_markdown(),
        doc_name=doc_name,
        docling_json=docling_data,
        document_id=identity.document_id,
        previous_structure=previous_structure,
    )
    tree["source_fingerprint"] = identity.fingerprint
    identity.document_dir.mkdir(parents=True, exist_ok=True)
    identity.structure_path.write_text(
        json.dumps(tree, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return identity.structure_path


@router.post("/documents/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receive, validate, store, and process an uploaded PDF.
    """

    try:
        # 1. Validate uploaded PDF
        content = await validate_pdf_upload(file)

        # 2. Resolve persistent identity and its storage paths.
        identity = await run_in_threadpool(IDENTITY_STORE.resolve, content)
        document_id = identity.document_id
        document_dir = identity.document_dir
        pdf_path = document_dir / "original.pdf"
        json_path = document_dir / "structured_document.json"
        document_dir.mkdir(parents=True, exist_ok=True)

        # 3. Save original PDF
        pdf_path.write_bytes(content)

        # 4. Convert PDF using Docling
        document = await run_in_threadpool(
            process_pdf,
            pdf_path
        )

        # 5. Export and save structured JSON
        await run_in_threadpool(
            save_structured_document,
            document,
            json_path
        )

        # 6. Extract and persist the stable retrieval structure.
        structure_path = await run_in_threadpool(
            _save_pageindex_structure,
            document,
            identity,
            file.filename or pdf_path.name,
        )

        # Index only after the complete retrieval structure has been persisted.
        await run_in_threadpool(_index_or_raise, structure_path)

        # 7. Return document information
        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "size": len(content),
            "status": "processed",
            "files": {
                "original_pdf": str(pdf_path),
                "structured_json": str(json_path),
                "structure_json": str(structure_path),
            }
        }

    except HTTPException:
        # Preserve the original PDF if processing fails.
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {str(exc)}"
        )

    finally:
        await file.close()

logger = logging.getLogger(__name__)

async def download_arxiv_pdf(pdf_url: str) -> bytes:
    """
    Download an arXiv PDF with a timeout and maximum size limit.
    """

    timeout = httpx.Timeout(30.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "GET",
            pdf_url,
            follow_redirects=True,
            headers={
                "User-Agent": "AgentResearcher/1.0"
            },
        ) as response:

            response.raise_for_status()

            # Only accept redirects to arXiv-controlled hosts.
            final_host = response.url.host

            if not (
                final_host == "arxiv.org"
                or final_host.endswith(".arxiv.org")
            ):
                raise ValueError("Unexpected download destination")

            # Check Content-Length when available.
            content_length = response.headers.get("content-length")

            if content_length:
                try:
                    if int(content_length) > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="PDF exceeds the 20 MB limit",
                        )
                except ValueError:
                    pass

            content = bytearray()

            async for chunk in response.aiter_bytes():
                content.extend(chunk)

                if len(content) > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF exceeds the 20 MB limit",
                    )

    # Verify that the downloaded content is actually a PDF.
    if not content or b"%PDF-" not in content[:1024]:
        raise ValueError("Downloaded content is not a valid PDF")

    return bytes(content)


@router.post("/documents/arxiv")
async def import_arxiv_paper(url: str = Query(...)):
    """
    Import an arXiv paper, download its PDF,
    process it with Docling, and save its structured JSON.
    """

    # 1. Validate and normalize the arXiv URL.
    try:
        pdf_url = normalize_arxiv_url(url)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    # Extract the arXiv paper ID from the canonical URL.
    paper_id = pdf_url.rstrip("/").split("/")[-1]

    # 2. Download the PDF.
    try:
        content = await download_arxiv_pdf(pdf_url)

    except HTTPException:
        raise

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Timed out while downloading the arXiv paper",
        )

    except httpx.HTTPStatusError as exc:
        status_code = (
            404 if exc.response.status_code == 404 else 502
        )
        raise HTTPException(
            status_code=status_code,
            detail="Unable to retrieve the PDF from arXiv",
        )

    except (httpx.RequestError, ValueError):
        logger.exception("Failed to download arXiv PDF")
        raise HTTPException(
            status_code=502,
            detail="Failed to download a valid PDF from arXiv",
        )

    # 3. Resolve persistent identity and storage paths.
    identity = await run_in_threadpool(IDENTITY_STORE.resolve, content)
    document_id = identity.document_id
    document_dir = identity.document_dir
    pdf_path = document_dir / "original.pdf"
    json_path = document_dir / "structured_document.json"

    try:
        document_dir.mkdir(parents=True, exist_ok=True)

        # 4. Save original PDF.
        await run_in_threadpool(pdf_path.write_bytes, content)

        # 5. Process PDF using Docling.
        document = await run_in_threadpool(
            process_pdf,
            pdf_path,
        )

        # 6. Export structured document as JSON.
        await run_in_threadpool(
            save_structured_document,
            document,
            json_path,
        )

        # 7. Extract and persist the stable retrieval structure.
        structure_path = await run_in_threadpool(
            _save_pageindex_structure,
            document,
            identity,
            f"{paper_id}.pdf",
        )

        # Reprocessing identical PDF bytes reuses the same identity, and
        # build_index replaces only that document's records.
        await run_in_threadpool(_index_or_raise, structure_path)

        # 8. Return document information.
        return {
            "success": True,
            "document_id": document_id,
            "source": "arxiv",
            "paper_id": paper_id,
            "filename": f"{paper_id}.pdf",
            "size": len(content),
            "status": "processed",
            "files": {
                "original_pdf": str(pdf_path),
                "structured_json": str(json_path),
                "structure_json": str(structure_path),
            },
        }

    except Exception:
        logger.exception(
            "Failed to process arXiv paper %s",
            paper_id,
        )

        # Keep the original PDF if Docling processing fails.
        raise HTTPException(
            status_code=500,
            detail="PDF downloaded, but document processing or indexing failed",
        )


@router.post("/query")
async def query_research(request: QueryRequest):
    """Route a query and execute the existing pipeline in document queues."""
    if not request.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be blank.")

    from .pipeline import run_pipeline

    try:
        routed = await run_in_threadpool(
            route_query,
            request.query,
            RETRIEVAL_DB_PATH,
            document_ids=request.document_ids,
        )
    except InvalidDocumentSelection as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EvaluationProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval is unavailable: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Research query pipeline failed")
        raise HTTPException(
            status_code=500,
            detail="Research query routing failed.",
        ) from exc

    query_id = str(uuid.uuid4())
    selected_document_ids = routed["selected_document_ids"]
    registry_by_id = {
        item["document_id"]: item
        for item in routed["registry"]
    }

    if not selected_document_ids:
        return {
            "query_id": query_id,
            "query": request.query,
            "routing": {
                "mode": routed["mode"],
                "selected_document_ids": [],
                "document_scores": routed["document_scores"],
            },
            "queue": None,
            "retrieval": routed["routing_retrieval"],
            "expanded_contexts": [],
            "evaluation": None,
            "loaded_documents": [],
            "document_results": [],
            "status": "no_documents",
        }

    def run_document(entry: DocumentQueueEntry):
        return run_pipeline(
            entry.query,
            llm_call=_openai_evaluator_call,
            db_path=RETRIEVAL_DB_PATH,
            document_ids=[entry.document_id],
            progress_callback=lambda stage, details: DOCUMENT_WORK_QUEUE.update_progress(
                query_id, entry.document_id, stage, details
            ),
        )

    try:
        futures = DOCUMENT_WORK_QUEUE.submit_query(
            query_id,
            request.query,
            selected_document_ids,
            run_document,
        )
    except Exception as exc:
        logger.exception("Could not enqueue research query")
        raise HTTPException(status_code=500, detail="Could not enqueue query work.") from exc

    async def collect_document(document_id: str, future):
        try:
            result = await asyncio.wait_for(
                asyncio.wrap_future(future),
                timeout=QUERY_DOCUMENT_TIMEOUT_SECONDS,
            )
            return {
                "document_id": document_id,
                "status": "completed",
                "result": result,
            }
        except asyncio.TimeoutError:
            future.cancel()
            DOCUMENT_WORK_QUEUE.mark_timed_out(query_id, document_id)
            return {
                "document_id": document_id,
                "status": "timed_out",
                "error": {
                    "type": "TimeoutError",
                    "message": "Document processing exceeded the request timeout.",
                    "stage": (DOCUMENT_WORK_QUEUE.snapshot(query_id) or {})
                    .get("documents", {}).get(document_id, {}).get("stage"),
                },
                "partial_result": (DOCUMENT_WORK_QUEUE.snapshot(query_id) or {})
                .get("documents", {}).get(document_id, {}).get("diagnostics", {}),
            }
        except asyncio.CancelledError:
            future.cancel()
            raise
        except Exception as exc:
            partial_result = getattr(exc, "partial_result", {})
            cause = getattr(exc, "cause", None)
            diagnostics = _serialize_evaluation_diagnostics(
                getattr(cause, "diagnostics", None)
            )
            error = {
                "type": type(getattr(exc, "cause", exc)).__name__,
                "message": str(exc),
                "stage": getattr(exc, "stage", None),
                "provider_error_type": getattr(cause, "provider_error_type", None),
            }
            if isinstance(cause, EvaluationValidationError):
                error["message"] = "Evidence evaluation validation failed."
            if diagnostics is not None:
                error["diagnostics"] = diagnostics
            return {
                "document_id": document_id,
                "status": "failed",
                "error": error,
                "partial_result": partial_result,
            }

    try:
        document_results = await asyncio.gather(*(
            collect_document(document_id, future)
            for document_id, future in futures.items()
        ))
    except asyncio.CancelledError:
        for future in futures.values():
            future.cancel()
        raise

    successful = [
        item for item in document_results
        if item["status"] == "completed"
    ]
    pipeline_results = [item["result"] for item in successful]
    partial_results = [item.get("partial_result", {}) for item in document_results
                       if item["status"] != "completed"]
    available_results = pipeline_results + partial_results
    merged_retrievals = [result["retrieval"] for result in available_results
                         if result.get("retrieval")]
    merged_evidence = [
        hit
        for result in merged_retrievals
        for hit in result.get("evidence", [])
    ]
    merged_navigation = [
        hit
        for result in merged_retrievals
        for hit in result.get("navigation_hits", [])
    ]
    successful_documents = [
        registry_by_id[item["document_id"]]
        for item in successful
    ]
    retrieval = {
        "query": request.query,
        "retrieval_status": "found" if merged_evidence else "not_found",
        "status_note": (
            "Status only indicates whether source blocks were retrieved. "
            "It does not establish that they fully answer the question."
        ),
        "evidence": merged_evidence,
        "navigation_hits": merged_navigation,
        "documents": [registry_by_id[item["document_id"]]
                      for item in document_results
                      if item["status"] == "completed" or item.get("partial_result", {}).get("retrieval")],
    }
    if merged_retrievals:
        retrieval["index_metadata"] = merged_retrievals[0].get("index_metadata", {})

    evaluations = {
        item["document_id"]: item["result"].get("evaluation")
        for item in successful
    }
    if len(selected_document_ids) == 1:
        evaluation = evaluations.get(selected_document_ids[0])
    else:
        evaluation = {"by_document": evaluations}

    failure_count = len(document_results) - len(successful)
    status = (
        "completed" if failure_count == 0
        else "partial" if successful
        else "failed"
    )
    queue_snapshot = DOCUMENT_WORK_QUEUE.snapshot(query_id)

    response_payload = {
        "query_id": query_id,
        "query": request.query,
        "routing": {
            "mode": routed["mode"],
            "selected_document_ids": selected_document_ids,
            "document_scores": routed["document_scores"],
        },
        "queue": queue_snapshot,
        "status": status,
        "retrieval": retrieval,
        "expanded_contexts": [
            context
            for result in available_results
            for context in result.get("expanded_contexts", [])
        ],
        "evaluation": evaluation,
        "loaded_documents": list(dict.fromkeys(
            document_id
            for result in available_results
            for document_id in result.get("loaded_documents", [])
        )),
        "document_results": document_results,
    }
    response_status = (
        200 if status == "completed"
        else 207 if status == "partial"
        else 503
    )
    return JSONResponse(content=response_payload, status_code=response_status)
