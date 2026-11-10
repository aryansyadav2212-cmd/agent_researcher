import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import requests
from docling.document_converter import DocumentConverter


OUTPUT_DIR = Path("data/papers")
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def extract_arxiv_id(url: str) -> str:
    """Extract and validate an arXiv paper ID from an abs/pdf URL."""

    parsed = urlparse(url)

    if parsed.scheme != "https":
        raise ValueError("Only HTTPS arXiv URLs are supported.")

    if parsed.hostname not in {"arxiv.org", "www.arxiv.org"}:
        raise ValueError("URL must belong to arxiv.org.")

    path = parsed.path.strip("/")

    if path.startswith("abs/"):
        paper_id = path.removeprefix("abs/")
    elif path.startswith("pdf/"):
        paper_id = path.removeprefix("pdf/")
    else:
        raise ValueError("Expected an arXiv /abs/ or /pdf/ URL.")

    # Remove optional .pdf extension.
    if paper_id.endswith(".pdf"):
        paper_id = paper_id[:-4]

    # Support modern IDs and legacy category-based IDs.
    pattern = r"^(?:\d{4}\.\d{4,5}|[a-zA-Z-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?$"

    if not re.fullmatch(pattern, paper_id):
        raise ValueError(f"Invalid arXiv paper ID: {paper_id}")

    return paper_id


def download_pdf(paper_id: str, destination: Path) -> None:
    """Download and preserve the original arXiv PDF."""

    pdf_url = f"https://arxiv.org/pdf/{paper_id}"

    headers = {
        "User-Agent": "ResearchAgent/0.1 (academic document processing)",
        "Accept": "application/pdf",
    }

    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(
        pdf_url,
        headers=headers,
        stream=True,
        timeout=(15, 120),
    ) as response:

        response.raise_for_status()

        total_size = 0

        with destination.open("wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if not chunk:
                    continue

                total_size += len(chunk)

                if total_size > MAX_FILE_SIZE:
                    raise ValueError("PDF exceeds the 100 MB size limit.")

                file.write(chunk)

    # Make sure we actually downloaded a PDF.
    with destination.open("rb") as file:
        if not file.read(5).startswith(b"%PDF-"):
            destination.unlink(missing_ok=True)
            raise ValueError("Downloaded content is not a valid PDF.")


def process_paper(paper_id: str, pdf_path: Path) -> None:
    """Convert PDF into Docling JSON and Markdown."""

    output_dir = pdf_path.parent

    json_path = output_dir / "document.json"
    markdown_path = output_dir / "document.md"

    print("\nProcessing PDF with Docling...")

    converter = DocumentConverter()

    result = converter.convert(str(pdf_path))

    document = result.document

    if document is None:
        raise RuntimeError(f"Docling conversion failed: {result.status}")

    # Export structured JSON.
    structured_data = document.export_to_dict()

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            structured_data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # Export Markdown.
    markdown = document.export_to_markdown()

    markdown_path.write_text(markdown, encoding="utf-8")

    # Release input resources after exporting.
    if result.input is not None:
        result.input.release()

    print("\nProcessing completed!")
    print(f"PDF:      {pdf_path}")
    print(f"JSON:     {json_path}")
    print(f"Markdown: {markdown_path}")
    print(f"Status:   {result.status}")


def main():
    parser = argparse.ArgumentParser(
        description="Download an arXiv paper and export it using Docling."
    )

    parser.add_argument(
        "url",
        help="arXiv abstract or PDF URL",
    )

    args = parser.parse_args()

    try:
        paper_id = extract_arxiv_id(args.url)

        # Replace legacy ID slash with underscore for folder name.
        folder_name = paper_id.replace("/", "_")

        paper_dir = OUTPUT_DIR / folder_name
        pdf_path = paper_dir / "original.pdf"

        print(f"Paper ID: {paper_id}")

        # Avoid downloading the same paper again.
        if pdf_path.exists():
            print("Existing PDF found. Skipping download.")
        else:
            print("Downloading research paper...")
            download_pdf(paper_id, pdf_path)

        process_paper(paper_id, pdf_path)

    except Exception as error:
        print(f"\nError: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()