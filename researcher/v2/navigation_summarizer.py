from __future__ import annotations

import argparse
import json
import hashlib
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

# Load OPENAI_API_KEY from a local .env when python-dotenv is installed.
# Exported environment variables continue to work without python-dotenv.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if TYPE_CHECKING:
    from openai import OpenAI

SUMMARY_VERSION = "1.1"
DEFAULT_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """You create concise navigation summaries for a research-paper section tree.
Your summaries help a retrieval agent decide which section to inspect next.

Rules:
- Use ONLY the supplied section title, direct content excerpts, and child summaries.
- Do not add facts, interpretations, results, or terminology that are not supported by the input.
- Describe the information covered, not whether the section is good or important.
- Treat direct_content_excerpt as first-class evidence. If it is non-empty, the summary MUST preserve its central subject, claim, method, or purpose; do not let child topics crowd it out.
- Also cover the topics represented by the immediate children. Each child summary is already a recursively generated summary of that child's entire subtree.
- When both direct content and children exist, synthesize them: explain the node's own contribution and how its child topics develop or specialize it. Do not merely concatenate child summaries.
- When direct content is empty, summarize the descendants as the node's subtree overview without inventing parent-level claims.
- Be specific enough to distinguish this section from neighboring sections.
- Keep each summary to 1-2 compact sentences (ideally 35-60 words; never exceed 75 words).
- Return 3-6 short key_items: concepts, methods, datasets, questions, or claims useful for routing.
- Do not repeat the section title as a key_item.
- If the supplied evidence is insufficient, say so briefly rather than guessing.

Return valid JSON only, with this exact shape:
{"summaries":[{"node_id":"...","summary":"...","key_items":["...", "..."]}]}"""


def _walk_nodes(nodes: list[dict[str, Any]], depth: int = 1) -> Iterable[tuple[int, dict[str, Any]]]:
    for node in nodes:
        yield depth, node
        yield from _walk_nodes(node.get("children", []), depth + 1)


def _all_nodes_by_depth(tree: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for depth, node in _walk_nodes(tree.get("structure", [])):
        by_depth[depth].append(node)
    return dict(by_depth)


def _first_sentences(text: str, max_chars: int) -> str:
    """Return a compact, deterministic excerpt without changing source content."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    # Prefer a sentence boundary near the limit; otherwise cut at a word boundary.
    candidate = text[:max_chars]
    boundary = max(candidate.rfind(". "), candidate.rfind("? "), candidate.rfind("! "))
    if boundary >= int(max_chars * 0.55):
        candidate = candidate[: boundary + 1]
    else:
        candidate = candidate.rsplit(" ", 1)[0]
    return candidate.rstrip() + " …[excerpt]"


def _direct_content_excerpt(node: dict[str, Any], max_chars: int) -> str:
    """Build a bounded excerpt from direct blocks, retaining representation of each block."""
    blocks = node.get("content_blocks", [])
    if not blocks or max_chars <= 0:
        return ""

    # Allocate the remaining budget across remaining blocks. This gives later
    # paragraphs/captions a chance to appear instead of letting the first long
    # paragraph consume the entire excerpt.
    pieces: list[str] = []
    remaining = max_chars
    nonempty = [b for b in blocks if (b.get("text") or "").strip()]
    for index, block in enumerate(nonempty):
        slots_left = len(nonempty) - index
        label = f"[{block.get('type', 'content')}] "
        share = max(0, (remaining - len(label)) // slots_left)
        if share < 40:
            break
        excerpt = _first_sentences(block.get("text", ""), share)
        piece = label + excerpt
        if len(piece) > remaining:
            piece = piece[:remaining]
        pieces.append(piece)
        remaining -= len(piece) + 1  # newline separator
        if remaining <= 0:
            break
    return "\n".join(pieces)


def _node_payload(node: dict[str, Any], *, node_char_budget: int) -> dict[str, Any]:
    children = [
        {
            "title": child.get("title", ""),
            "summary": child.get("summary") or "",
            "key_items": child.get("key_items", []),
        }
        for child in node.get("children", [])
    ]
    return {
        "node_id": node.get("node_id", ""),
        "title": node.get("title", ""),
        "section_number": node.get("number"),
        "direct_content_excerpt": _direct_content_excerpt(node, node_char_budget),
        "children": children,
    }


def _pack_nodes(
    nodes: list[dict[str, Any]], *, node_char_budget: int, batch_char_budget: int
) -> list[list[dict[str, Any]]]:
    """Pack nodes into request batches using character budgets as a token-cost proxy."""
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_size = 0

    for node in nodes:
        payload = _node_payload(node, node_char_budget=node_char_budget)
        size = len(json.dumps(payload, ensure_ascii=False))
        if current and current_size + size > batch_char_budget:
            batches.append(current)
            current = []
            current_size = 0
        current.append(payload)
        current_size += size
    if current:
        batches.append(current)
    return batches


def _summary_fingerprint(payload: dict[str, Any], model: str) -> str:
    """Identify the exact evidence/model version used to create a summary.

    Because child summaries are part of the payload, a changed descendant
    summary naturally invalidates every ancestor summary on the next run.
    """
    material = {
        "summary_version": SUMMARY_VERSION,
        "model": model,
        "payload": payload,
    }
    encoded = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _fallback_summary(node: dict[str, Any]) -> None:
    """Avoid an API call for a truly empty leaf; never invent its contents."""
    title = node.get("title") or "Untitled section"
    node["summary"] = f"Section titled '{title}'; no extractable text or child-section content was available to summarize."
    node["key_items"] = []
    node["summary_source"] = "deterministic_fallback"


def _summarize_batch(
    client: OpenAI,
    *,
    model: str,
    batch: list[dict[str, Any]],
    max_output_tokens: int,
) -> dict[str, dict[str, Any]]:
    user_payload = {
        "task": "Create one navigation summary for each supplied node.",
        "nodes": batch,
    }
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_completion_tokens=max_output_tokens,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("OpenAI returned an empty summary response")
    parsed = json.loads(content)
    summaries = parsed.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Expected JSON object with a 'summaries' array")

    expected_ids = {str(item["node_id"]) for item in batch}
    result: dict[str, dict[str, Any]] = {}
    for item in summaries:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id", ""))
        summary = item.get("summary")
        key_items = item.get("key_items", [])
        if node_id not in expected_ids or not isinstance(summary, str) or not summary.strip():
            continue
        if not isinstance(key_items, list):
            key_items = []
        result[node_id] = {
            "summary": summary.strip(),
            "key_items": [str(value).strip() for value in key_items if str(value).strip()][:6],
        }

    missing = expected_ids - result.keys()
    if missing:
        raise ValueError(f"Model omitted summaries for node IDs: {', '.join(sorted(missing))}")
    return result


def generate_navigation_summaries(
    tree: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    node_char_budget: int = 4500,
    batch_char_budget: int = 12000,
    max_output_tokens: int = 1200,
    force: bool = False,
    dry_run: bool = False,
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate recursive navigation summaries, bottom-up, while preserving source blocks.

    Each node is summarized from (1) its own bounded direct-content excerpt and
    (2) summaries of its immediate children. Since children are processed first,
    each child summary already represents that child's descendants.

    A fingerprint makes this incremental: if direct evidence, a child summary,
    the model, or the prompt version changes, the affected node and its ancestors
    are regenerated. Unchanged nodes incur no API calls.
    """
    by_depth = _all_nodes_by_depth(tree)
    if not by_depth:
        tree["document_summary"] = "No section headings were extracted from this document."
        return tree

    # Load the client only when a real request is required. Dry-run never needs
    # an API key and never makes a network request.
    api_client = client
    client_loaded = client is not None
    total_summarized = 0
    planned_batches: list[dict[str, Any]] = []

    # Process deepest nodes first. At each depth, all child summaries are ready.
    for depth in sorted(by_depth, reverse=True):
        candidates: list[dict[str, Any]] = []
        fingerprints: dict[str, str] = {}
        nodes_by_id = {str(node.get("node_id", "")): node for node in by_depth[depth]}

        for node in by_depth[depth]:
            has_content = any((b.get("text") or "").strip() for b in node.get("content_blocks", []))
            has_children = bool(node.get("children"))

            if not has_content and not has_children:
                fallback = (
                    f"Section titled '{node.get('title') or 'Untitled section'}'; "
                    "no extractable text or child-section content was available to summarize."
                )
                if force or node.get("summary") != fallback:
                    node["summary"] = fallback
                    node["key_items"] = []
                    node["summary_source"] = "deterministic_fallback"
                    node["summary_input_hash"] = _summary_fingerprint(
                        {"node_id": node.get("node_id", ""), "title": node.get("title", ""), "empty": True}, model
                    )
                continue

            payload = _node_payload(node, node_char_budget=node_char_budget)
            fingerprint = _summary_fingerprint(payload, model)
            unchanged = (
                not force
                and bool(node.get("summary"))
                and node.get("summary_input_hash") == fingerprint
            )
            if unchanged:
                continue

            candidates.append(node)
            fingerprints[str(node.get("node_id", ""))] = fingerprint

        if not candidates:
            continue

        batches = _pack_nodes(
            candidates,
            node_char_budget=node_char_budget,
            batch_char_budget=batch_char_budget,
        )
        for batch in batches:
            node_ids = [str(item.get("node_id", "")) for item in batch]
            planned_batches.append({"depth": depth, "node_count": len(batch), "node_ids": node_ids})

            if dry_run:
                continue

            if not client_loaded:
                from openai import OpenAI
                if not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY is not set. Export it before running without --dry-run.")
                api_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=1)
                client_loaded = True

            summaries = _summarize_batch(
                api_client,
                model=model,
                batch=batch,
                max_output_tokens=max_output_tokens,
            )
            for node_id, summary_data in summaries.items():
                node = nodes_by_id[node_id]
                node["summary"] = summary_data["summary"]
                node["key_items"] = summary_data["key_items"]
                node["summary_source"] = "openai"
                node["summary_input_hash"] = fingerprints[node_id]
                total_summarized += 1

    if dry_run:
        tree["summary_generation_plan"] = {
            "model": model,
            "request_count": len(planned_batches),
            "batches": planned_batches,
            "note": (
                "No OpenAI requests were made. Character budgets approximate input size; "
                "they are not exact token counts. Parent payloads contain direct text plus "
                "immediate child summaries, which recursively represent descendants."
            ),
        }
        return tree

    # Generate/update the document-level description only when its root evidence changes.
    if tree.get("structure"):
        root_payload = [
            {"title": node.get("title", ""), "summary": node.get("summary", ""), "key_items": node.get("key_items", [])}
            for node in tree.get("structure", [])
        ]
        document_fingerprint = _summary_fingerprint({"root_sections": root_payload}, model)
        should_update_document = force or not tree.get("document_summary") or tree.get("document_summary_input_hash") != document_fingerprint
        if should_update_document:
            if not client_loaded:
                from openai import OpenAI
                if not os.environ.get("OPENAI_API_KEY"):
                    raise RuntimeError("OPENAI_API_KEY is not set. Export it before running without --dry-run.")
                api_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=1)
                client_loaded = True
            response = api_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write a concise 2-3 sentence document description for a retrieval agent. "
                            "Use only the supplied root-section summaries and key items. Cover the paper's "
                            "overall subject and its main components. Do not invent facts. "
                            'Return JSON only: {"document_summary": "..."}.'
                        ),
                    },
                    {"role": "user", "content": json.dumps({"sections": root_payload}, ensure_ascii=False, separators=(",", ":"))},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_completion_tokens=180,
            )
            content = response.choices[0].message.content or "{}"
            tree["document_summary"] = json.loads(content).get("document_summary", "")
            tree["document_summary_input_hash"] = document_fingerprint

    tree["summary_metadata"] = {
        "version": SUMMARY_VERSION,
        "model": model,
        "strategy": "recursive_bottom_up_direct_content_plus_descendant_summaries",
        "summarized_nodes_this_run": total_summarized,
        "node_char_budget": node_char_budget,
        "batch_char_budget": batch_char_budget,
        "max_output_tokens_per_batch": max_output_tokens,
        "incremental_cache": "summary_input_hash",
    }
    return tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate low-cost bottom-up navigation summaries for a document tree")
    parser.add_argument("input", type=Path, help="structure.json produced by markdown_extractor.py")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path (default: <input>.summarized.json)")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--node-char-budget", type=int, default=4500)
    parser.add_argument("--batch-char-budget", type=int, default=12000)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--force", action="store_true", help="Regenerate summaries even when a node already has one")
    parser.add_argument("--dry-run", action="store_true", help="Show planned batches without calling OpenAI")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    result = generate_navigation_summaries(
        data,
        model=args.model,
        node_char_budget=args.node_char_budget,
        batch_char_budget=args.batch_char_budget,
        max_output_tokens=args.max_output_tokens,
        force=args.force,
        dry_run=args.dry_run,
    )
    output = args.output or args.input.with_name(f"{args.input.stem}.summarized.json")
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.dry_run:
        plan = result.get("summary_generation_plan", {})
        print(f"Dry run: {plan.get('request_count', 0)} OpenAI batch requests would be made.")
    else:
        meta = result.get("summary_metadata", {})
        print(f"Summarized {meta.get('summarized_nodes_this_run', 0)} nodes using {args.model}.")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
