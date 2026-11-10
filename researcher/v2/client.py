
"""
Main entry point for document indexing, retrieval,
and conversation flows.
"""

import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
import logging
import uuid
from pathlib import Path
import json

import httpx
from fastapi import Query, HTTPException
from starlette.concurrency import run_in_threadpool
from .document_utils import (
    validate_pdf_upload,
    process_pdf,
    save_structured_document,
    normalize_arxiv_url,
    MAX_FILE_SIZE,
)

router = APIRouter()

DOCUMENTS_DIR = Path("storage/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/documents/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receive, validate, store, and process an uploaded PDF.
    """

    document_id = str(uuid.uuid4())

    document_dir = DOCUMENTS_DIR / document_id
    document_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = document_dir / "original.pdf"
    json_path = document_dir / "structured_document.json"

    try:
        # 1. Validate uploaded PDF
        content = await validate_pdf_upload(file)

        # 2. Save original PDF
        pdf_path.write_bytes(content)

        # 3. Convert PDF using Docling
        document = await run_in_threadpool(
            process_pdf,
            pdf_path
        )

        # 4. Export and save structured JSON
        await run_in_threadpool(
            save_structured_document,
            document,
            json_path
        )

        # 5. Return document information
        return {
            "success": True,
            "document_id": document_id,
            "filename": file.filename,
            "size": len(content),
            "status": "processed",
            "files": {
                "original_pdf": str(pdf_path),
                "structured_json": str(json_path)
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

    # 3. Create storage paths.
    document_id = str(uuid.uuid4())

    document_dir = Path("storage/documents") / document_id
    pdf_path = document_dir / "original.pdf"
    json_path = document_dir / "structured_document.json"

    try:
        document_dir.mkdir(parents=True, exist_ok=False)

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

        # 7. Return document information.
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
            detail="PDF downloaded, but document processing failed",
        )