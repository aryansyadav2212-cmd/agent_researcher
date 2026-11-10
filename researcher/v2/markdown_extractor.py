from __future__ import annotations

import html
import json
import re
import uuid
from pathlib import Path
from typing import Any

# Handles both normal Markdown headings and Docling's **## Heading** form.
_HEADING_RE = re.compile(r"^\s*(?:\*\*)?\s*(#{1,6})\s+(.+?)\s*(?:\*\*)?\s*$")
_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s*(.*)$")
_KNOWN_UNNUMBERED_SECTIONS = {
    "abstract", "introduction", "background", "related work", "methodology",
    "methods", "approach", "experiments", "results", "discussion",
    "conclusion", "conclusions", "acknowledgements", "acknowledgments",
    "references", "appendix", "supplementary material",
}


def _clean_heading(value: str) -> str:
    value = html.unescape(value).strip()
    # Remove common trailing Markdown artifacts without altering meaningful punctuation.
    return value.strip("* ")


def _heading_number(title: str) -> tuple[int | None, str | None]:
    match = _NUMBER_RE.match(title)
    if not match:
        return None, None
    number = match.group(1)
    remainder = match.group(2).strip()
    return number.count(".") + 1, number if not remainder else number


def _normalise_title(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value)).strip().casefold()


def _docling_heading_levels(docling_json: dict[str, Any] | None) -> dict[str, int]:
    """Build a best-effort title -> level map from Docling's exported dictionary."""
    result: dict[str, int] = {}
    if not docling_json:
        return result

    for item in docling_json.get("texts", []):
        if item.get("label") != "section_header":
            continue
        title = _normalise_title(item.get("text") or item.get("orig") or "")
        level = item.get("level")
        if title and isinstance(level, int) and level > 0:
            result[title] = level
    return result


def _split_blocks(markdown: str) -> list[dict[str, Any]]:
    """Split Markdown into non-empty blocks while retaining 1-based source lines."""
    lines = markdown.splitlines()
    blocks: list[dict[str, Any]] = []
    current: list[str] = []
    start_line: int | None = None

    def flush(end_line: int) -> None:
        nonlocal current, start_line
        if current:
            text = "\n".join(current).strip()
            if text:
                blocks.append({
                    "block_id": f"blk_{len(blocks) + 1:06d}",
                    "type": "content",
                    "text": text,
                    "start_line": start_line,
                    "end_line": end_line,
                })
        current = []
        start_line = None

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            flush(line_number - 1)
        else:
            if start_line is None:
                start_line = line_number
            current.append(line)
    flush(len(lines))
    return blocks


def _classify_content(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        return "code"
    if stripped.startswith(("<!-- image", "![", "<img")):
        return "image"
    lines = stripped.splitlines()
    if len(lines) >= 2 and all("|" in line for line in lines[:2]):
        return "table"
    if re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)", stripped):
        return "list"
    if stripped.startswith(">"):
        return "blockquote"
    return "paragraph"


def _detect_heading(block_text: str) -> dict[str, Any] | None:
    # Docling's heading is normally one line. Multi-line blocks are not headings.
    if "\n" in block_text:
        return None
    match = _HEADING_RE.match(block_text)
    if not match:
        return None
    title = _clean_heading(match.group(2))
    if not title:
        return None
    markdown_level = len(match.group(1))
    number_level, number = _heading_number(title)
    return {
        "title": title,
        "markdown_level": markdown_level,
        "number_level": number_level,
        "number": number,
    }


def _resolve_level(heading: dict[str, Any], docling_levels: dict[str, int]) -> tuple[int, str]:
    """Priority: explicit numbering, Docling level, then Markdown heading depth."""
    if heading["number_level"] is not None:
        return heading["number_level"], "numbering"
    docling_level = docling_levels.get(_normalise_title(heading["title"]))
    if docling_level is not None:
        return docling_level, "docling"
    if _normalise_title(heading["title"]) in _KNOWN_UNNUMBERED_SECTIONS:
        return 1, "known_section_fallback"
    return heading["markdown_level"], "markdown"


def extract_markdown_structure(
    markdown: str,
    *,
    doc_name: str,
    docling_json: dict[str, Any] | None = None,
    doc_title: str | None = None,
) -> dict[str, Any]:
    """Extract a deterministic heading tree and direct content from Docling Markdown.

    `start_line`/`end_line` are 1-based and inclusive. `content_blocks` contains
    only content directly associated with that heading, never descendant content.
    """
    blocks = _split_blocks(markdown)
    docling_levels = _docling_heading_levels(docling_json)

    # First pass: identify heading blocks and resolve their levels.
    heading_indices: list[int] = []
    for index, block in enumerate(blocks):
        detected = _detect_heading(block["text"])
        if detected:
            level, source = _resolve_level(detected, docling_levels)
            block["type"] = "heading"
            block["heading"] = {**detected, "level": level, "level_source": source}
            heading_indices.append(index)
        else:
            block["type"] = "content"

    # Identify the document title: first non-numbered heading, unless it is a
    # conventional section heading such as Abstract. Caller can override it.
    detected_title = None
    if doc_title:
        detected_title = doc_title.strip()
    elif heading_indices:
        first_heading = blocks[heading_indices[0]]["heading"]
        if (
            first_heading["number_level"] is None
            and _normalise_title(first_heading["title"]) not in _KNOWN_UNNUMBERED_SECTIONS
        ):
            detected_title = first_heading["title"]

    # Build nodes in document order. Preamble blocks are preserved separately.
    root_nodes: list[dict[str, Any]] = []
    preamble: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    current_node: dict[str, Any] | None = None
    heading_position = 0

    for index, block in enumerate(blocks):
        if block["type"] == "heading":
            heading = block["heading"]
            is_document_title = (
                detected_title is not None
                and heading_position == 0
                and _normalise_title(heading["title"]) == _normalise_title(detected_title)
            )
            heading_position += 1
            if is_document_title:
                continue

            level = heading["level"]
            while stack and stack[-1]["level"] >= level:
                stack.pop()

            node = {
                "node_id": f"node_{uuid.uuid4().hex[:12]}",
                "title": heading["title"],
                "number": heading["number"],
                "level": level,
                "level_source": heading["level_source"],
                "start_line": block["start_line"],
                "end_line": block["end_line"],
                "content_blocks": [],
                "children": [],
                "summary": None,
                "key_items": [],
            }

            if stack:
                stack[-1]["children"].append(node)
            else:
                root_nodes.append(node)
            stack.append(node)
            current_node = node
        else:
            content_block = {
                "block_id": block["block_id"],
                "type": _classify_content(block["text"]),
                "text": block["text"],
                "start_line": block["start_line"],
                "end_line": block["end_line"],
            }
            if stack:
                stack[-1]["content_blocks"].append(content_block)
                stack[-1]["end_line"] = block["end_line"]
            else:
                preamble.append(content_block)

    # Finalize section spans: a parent's range includes its descendants, while
    # content_blocks still contain only the parent's direct content.
    def finalize_spans(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            finalize_spans(node["children"])
            if node["children"]:
                node["end_line"] = max(
                    node["end_line"],
                    max(child["end_line"] for child in node["children"]),
                )

    finalize_spans(root_nodes)

    return {
        "document_id": f"doc_{uuid.uuid4().hex[:12]}",
        "doc_name": doc_name,
        "doc_title": detected_title,
        "preamble": preamble,
        "structure": root_nodes,
        "source_format": "docling_markdown",
        "line_numbering": "1-based inclusive",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract a heading tree from Docling Markdown")
    parser.add_argument("markdown_path", type=Path)
    parser.add_argument("--doc-name", default=None)
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--doc-title", default=None)
    parser.add_argument("--output", type=Path, default=Path("structure.json"))
    args = parser.parse_args()

    markdown_text = args.markdown_path.read_text(encoding="utf-8")
    docling_data = json.loads(args.json_path.read_text(encoding="utf-8")) if args.json_path else None
    result = extract_markdown_structure(
        markdown_text,
        doc_name=args.doc_name or args.markdown_path.stem,
        docling_json=docling_data,
        doc_title=args.doc_title,
    )
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Extracted {len(result['structure'])} root sections -> {args.output}")
