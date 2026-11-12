from __future__ import annotations

import html
import json
import re
import uuid
from difflib import SequenceMatcher
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


def _new_identity(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _is_persistent_identity(value: Any, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and value.startswith(f"{prefix}_")
        and len(value) == len(prefix) + 33
        and all(character in "0123456789abcdef" for character in value[len(prefix) + 1 :])
    )


def _block_signature(block: dict[str, Any]) -> tuple[str, str]:
    return (str(block.get("type") or ""), str(block.get("text") or ""))


def _reuse_block_ids(
    previous_blocks: list[dict[str, Any]],
    current_blocks: list[dict[str, Any]],
) -> None:
    previous_signatures = [_block_signature(block) for block in previous_blocks]
    current_signatures = [_block_signature(block) for block in current_blocks]
    matcher = SequenceMatcher(
        a=previous_signatures,
        b=current_signatures,
        autojunk=False,
    )
    for previous_index, current_index, size in matcher.get_matching_blocks():
        for offset in range(size):
            previous = previous_blocks[previous_index + offset]
            current = current_blocks[current_index + offset]
            block_id = previous.get("block_id")
            if _is_persistent_identity(block_id, "blk"):
                current["block_id"] = block_id


def _node_disambiguator(node: dict[str, Any]) -> tuple[Any, ...]:
    return (
        tuple(_block_signature(block) for block in node.get("content_blocks", [])),
        tuple(
            (
                _normalise_title(child.get("title") or ""),
                str(child.get("number") or ""),
            )
            for child in node.get("children", [])
        ),
    )


def _match_sibling_nodes(
    previous_nodes: list[dict[str, Any]],
    current_nodes: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    previous_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    current_groups: dict[tuple[str, str, int], list[tuple[int, dict[str, Any]]]] = {}

    for node in previous_nodes:
        key = (
            _normalise_title(node.get("title") or ""),
            str(node.get("number") or ""),
            node.get("level", 0),
        )
        previous_groups.setdefault(key, []).append(node)
    for index, node in enumerate(current_nodes):
        key = (
            _normalise_title(node.get("title") or ""),
            str(node.get("number") or ""),
            node.get("level", 0),
        )
        current_groups.setdefault(key, []).append((index, node))

    matches: dict[int, dict[str, Any]] = {}
    for key, current_group in current_groups.items():
        previous_group = previous_groups.get(key, [])
        if len(current_group) == 1 and len(previous_group) == 1:
            matches[current_group[0][0]] = previous_group[0]
            continue

        previous_by_signature: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        current_by_signature: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = {}
        for node in previous_group:
            previous_by_signature.setdefault(_node_disambiguator(node), []).append(node)
        for index, node in current_group:
            current_by_signature.setdefault(_node_disambiguator(node), []).append((index, node))

        for signature, current_duplicates in current_by_signature.items():
            previous_duplicates = previous_by_signature.get(signature, [])
            if len(current_duplicates) == 1 and len(previous_duplicates) == 1:
                matches[current_duplicates[0][0]] = previous_duplicates[0]
    return matches


def _reuse_node_identities(
    previous_nodes: list[dict[str, Any]],
    current_nodes: list[dict[str, Any]],
) -> None:
    matches = _match_sibling_nodes(previous_nodes, current_nodes)
    for current_index, current in enumerate(current_nodes):
        previous = matches.get(current_index)
        if previous is None:
            continue

        previous_id = previous.get("node_id")
        if _is_persistent_identity(previous_id, "node"):
            current["node_id"] = previous_id

        _reuse_block_ids(
            previous.get("content_blocks", []),
            current.get("content_blocks", []),
        )
        _reuse_node_identities(
            previous.get("children", []),
            current.get("children", []),
        )


def _reuse_structure_identities(
    previous_structure: dict[str, Any] | None,
    current_structure: dict[str, Any],
) -> None:
    if not previous_structure:
        return
    _reuse_block_ids(
        previous_structure.get("preamble", []),
        current_structure["preamble"],
    )
    _reuse_node_identities(
        previous_structure.get("structure", []),
        current_structure["structure"],
    )


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
                    "block_id": _new_identity("blk"),
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
    document_id: str | None = None,
    previous_structure: dict[str, Any] | None = None,
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
                "node_id": _new_identity("node"),
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

    result = {
        "document_id": document_id
        or (previous_structure or {}).get("document_id")
        or _new_identity("doc"),
        "doc_name": doc_name,
        "doc_title": detected_title,
        "preamble": preamble,
        "structure": root_nodes,
        "source_format": "docling_markdown",
        "line_numbering": "1-based inclusive",
    }
    _reuse_structure_identities(previous_structure, result)
    return result


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
