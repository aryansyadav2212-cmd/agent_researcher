
from urllib.parse import urlparse
from io import BytesIO
from pathlib import Path
import re
import json
from fastapi import UploadFile, HTTPException
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter


# Configuration
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

# Initialize Docling converter
converter = DocumentConverter()


async def validate_pdf_upload(file: UploadFile) -> bytes:
    """Validate an uploaded PDF and return its binary content."""

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    allowed_types = {
        "application/pdf",
        "application/octet-stream",
    }

    if file.content_type not in allowed_types:
        raise HTTPException(400, "Invalid file content type")

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(400, "Uploaded file is empty")

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, "PDF exceeds the 20 MB limit")

    if b"%PDF-" not in content[:1024]:
        raise HTTPException(400, "Invalid PDF file")

    return content



def process_pdf(file_path: Path):
    """
    Convert a saved PDF into a structured DoclingDocument.
    """

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    if file_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    result = converter.convert(file_path)

    return result.document


def save_structured_document(document, output_path: Path) -> None:
    """
    Export a DoclingDocument to JSON and persist it.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    structured_data = document.export_to_dict()

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            structured_data,
            json_file,
            ensure_ascii=False,
            indent=2
        )

def normalize_arxiv_url(url: str) -> str:
    """
    Validate an arXiv URL and convert it into a canonical PDF URL.

    Accepts:
        https://arxiv.org/abs/2609.03874
        https://arxiv.org/pdf/2609.03874
        https://arxiv.org/abs/2609.03874v2
        https://arxiv.org/pdf/2609.03874v2.pdf

    Returns:
        https://arxiv.org/pdf/2609.03874
    """

    if not isinstance(url, str) or not url.strip():
        raise ValueError("URL must be a non-empty string")

    url = url.strip()
    parsed = urlparse(url)

    # Validate domain and scheme
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only HTTP or HTTPS URLs are supported")

    if parsed.hostname not in ("arxiv.org", "www.arxiv.org"):
        raise ValueError("Only arxiv.org URLs are supported")

    if parsed.port is not None:
        raise ValueError("Custom ports are not allowed")

    # Validate URL path
    match = re.fullmatch(
        r"/(?:abs|pdf)/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?/?",
        parsed.path
    )

    if not match:
        raise ValueError("Invalid or unsupported arXiv URL")

    paper_id = match.group(1)

    # Construct canonical PDF URL
    return f"https://arxiv.org/pdf/{paper_id}"
