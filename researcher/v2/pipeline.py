"""Orchestrate retrieval, context expansion, and evidence evaluation."""

from pathlib import Path
from typing import Any, Callable

from .context_expander import ContextExpander
from .evidence_evaluator import EvidenceEvaluator
from .structure_index import StructureIndex


_V2_DIR = Path(__file__).resolve().parent
DEFAULT_STRUCTURE_PATH = _V2_DIR / "structure.json"
DEFAULT_DB_PATH = _V2_DIR / "storage" / "retrieval" / "index.sqlite3"


def _expand_preamble_context(
    evidence: dict[str, Any],
    structure_index: StructureIndex,
    before: int,
    after: int,
) -> dict[str, Any]:
    """Adapt retrieval's synthetic preamble node to the expander context shape."""
    for name, value in (("before", before), ("after", after)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"'{name}' must be a non-negative integer.")

    blocks = structure_index.preamble
    block_id = evidence["content_block_id"]
    anchor_position = next(
        (
            position
            for position, block in enumerate(blocks)
            if block["block_id"] == block_id
        ),
        None,
    )
    if anchor_position is None:
        raise KeyError(f"Preamble block not found: {block_id}")

    start = max(0, anchor_position - before)
    end = min(len(blocks), anchor_position + after + 1)
    return {
        "node_id": "__preamble__",
        "anchor_block_id": block_id,
        "anchor_position": anchor_position,
        "window": {
            "start_position": start,
            "end_position": end - 1,
        },
        "blocks": [
            {
                **block,
                "position": position,
                "is_anchor": block["block_id"] == block_id,
            }
            for position, block in enumerate(blocks[start:end], start=start)
        ],
    }


def run_pipeline(
    query: str,
    *,
    llm_call: Callable[[str], str],
    structure_path: str | Path = DEFAULT_STRUCTURE_PATH,
    db_path: str | Path = DEFAULT_DB_PATH,
    top_k: int = 8,
    candidate_k: int = 30,
    before: int = 2,
    after: int = 2,
    retriever: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the existing retrieval components and return their structured results.

    ``llm_call`` is supplied by the caller because the evaluator intentionally
    does not define or create its own LLM client. ``retriever`` is an optional
    seam for callers and integration tests; by default the existing
    ``retrieval.search_index`` function is used.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")
    if not callable(llm_call):
        raise TypeError("llm_call must be callable.")

    if retriever is None:
        from .retrieval import search_index

        retriever = search_index

    retrieval_result = retriever(
        query,
        Path(db_path),
        top_k=top_k,
        candidate_k=candidate_k,
    )
    if not isinstance(retrieval_result, dict):
        raise TypeError("Retriever must return a dictionary.")

    evidence_items = retrieval_result.get("evidence")
    if not isinstance(evidence_items, list):
        raise ValueError("Retriever result must contain an evidence list.")

    structure_index = StructureIndex.from_file(structure_path)
    expanded_contexts = []
    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            raise ValueError("Each retrieved evidence item must be a dictionary.")
        if evidence.get("node_id") == "__preamble__":
            expanded_contexts.append(
                _expand_preamble_context(
                    evidence,
                    structure_index,
                    before,
                    after,
                )
            )
        else:
            expanded_contexts.append(
                ContextExpander(evidence, structure_index).expand_context(
                    before=before,
                    after=after,
                )
            )

    evaluation = EvidenceEvaluator(
        query=query,
        expanded_contexts=expanded_contexts,
        llm_call=llm_call,
    ).evaluate()

    return {
        "query": query,
        "retrieval": retrieval_result,
        "expanded_contexts": expanded_contexts,
        "evaluation": evaluation,
    }
