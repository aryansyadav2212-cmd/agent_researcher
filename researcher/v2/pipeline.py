"""Orchestrate multi-document retrieval, context expansion, and evaluation."""

from pathlib import Path
from typing import Any, Callable
import time

from .context_expander import ContextExpander
from .evidence_evaluator import EvidenceEvaluator
from .structure_index import StructureIndex


_V2_DIR = Path(__file__).resolve().parent
DEFAULT_STRUCTURE_PATH = _V2_DIR / "structure.json"
DEFAULT_DB_PATH = _V2_DIR / "storage" / "retrieval" / "index.sqlite3"


class PipelineStageError(RuntimeError):
    """Pipeline failure carrying completed stage data for API diagnostics."""

    def __init__(self, stage: str, cause: Exception, partial_result: dict[str, Any]):
        super().__init__(f"{stage}: {type(cause).__name__}: {cause}")
        self.stage = stage
        self.cause = cause
        self.partial_result = partial_result


def _validate_expansion_boundaries(before: int, after: int) -> None:
    for name, value in (("before", before), ("after", after)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"'{name}' must be a non-negative integer.")


def _expand_preamble_context(
    evidence: dict[str, Any],
    structure_index: StructureIndex,
    before: int,
    after: int,
) -> dict[str, Any]:
    """Expand context around a preamble block."""
    _validate_expansion_boundaries(before, after)

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
        "document_id": evidence.get("document_id"),
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


def _build_document_registry(
    retrieval_result: dict[str, Any],
    structure_path: str | Path,
    evidence_items: list[dict[str, Any]],
) -> dict[str, Path]:
    """Resolve document IDs to their corresponding structure files."""
    documents = retrieval_result.get("documents")

    if documents is None:
        # Backward compatibility for the previous single-document retriever.
        document_ids = {
            item.get("document_id")
            for item in evidence_items
            if item.get("document_id")
        }

        if len(document_ids) > 1:
            raise ValueError(
                "Multi-document evidence was returned without a documents "
                "registry. Update retrieval.py to return document metadata."
            )

        if len(document_ids) == 1:
            document_id = next(iter(document_ids))
            return {document_id: Path(structure_path).resolve()}

        # Legacy results without document_id.
        return {"__legacy__": Path(structure_path).resolve()}

    if not isinstance(documents, list):
        raise ValueError("Retriever documents field must be a list.")

    registry: dict[str, Path] = {}

    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Each document registry entry must be a dictionary.")

        document_id = document.get("document_id")
        document_structure_path = document.get("structure_path")

        if (
            not isinstance(document_id, str)
            or not document_id
            or document_id != document_id.strip()
        ):
            raise ValueError("Every registered document must have a document_id.")

        if not isinstance(document_structure_path, str) or not document_structure_path.strip():
            raise ValueError(
                f"Document {document_id!r} has no valid structure_path."
            )

        if document_id in registry:
            raise ValueError(
                f"Duplicate document_id in registry: {document_id}"
            )

        resolved_structure_path = Path(document_structure_path).resolve()
        if resolved_structure_path.exists() and resolved_structure_path.is_dir():
            raise ValueError(
                f"Document {document_id!r} structure_path points to a directory: "
                f"{resolved_structure_path}"
            )
        registry[document_id] = resolved_structure_path

    if evidence_items and not registry:
        raise ValueError(
            "The retriever returned evidence without registering any documents."
        )

    return registry


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
    document_ids: list[str] | None = None,
    progress_callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run multi-document retrieval, context expansion, and evaluation."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if not callable(llm_call):
        raise TypeError("llm_call must be callable.")

    if document_ids is not None:
        if not isinstance(document_ids, list) or not document_ids:
            raise ValueError("document_ids must be a non-empty list when supplied.")
        for index, document_id in enumerate(document_ids):
            if (
                not isinstance(document_id, str)
                or not document_id
                or document_id != document_id.strip()
            ):
                raise ValueError(
                    f"document_ids[{index}] must be a non-empty, trimmed string."
                )
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("document_ids must not contain duplicates.")

    _validate_expansion_boundaries(before, after)

    if retriever is None:
        from .retrieval import search_index

        retriever = search_index

    retrieval_kwargs = {
        "top_k": top_k,
        "candidate_k": candidate_k,
    }
    if document_ids is not None:
        retrieval_kwargs["document_ids"] = document_ids
    stage = "retrieval"
    stage_started = time.monotonic()
    partial_result: dict[str, Any] = {"query": query, "retrieval": None,
                                      "expanded_contexts": [], "evaluation": None,
                                      "loaded_documents": []}
    if progress_callback:
        progress_callback("retrieval", {"elapsed_seconds": 0.0})
    try:
        retrieval_result = retriever(query, Path(db_path), **retrieval_kwargs)
    except Exception as exc:
        raise PipelineStageError(stage, exc, partial_result) from exc
    partial_result["retrieval"] = retrieval_result
    if progress_callback:
        progress_callback("retrieval", {"elapsed_seconds": time.monotonic() - stage_started,
                                         "retrieval": retrieval_result})

    if not isinstance(retrieval_result, dict):
        raise TypeError("Retriever must return a dictionary.")

    evidence_items = retrieval_result.get("evidence")

    if not isinstance(evidence_items, list):
        raise ValueError("Retriever result must contain an evidence list.")

    navigation_hits = retrieval_result.get("navigation_hits", [])
    if not isinstance(navigation_hits, list):
        raise ValueError("Retriever navigation_hits field must be a list.")

    for evidence in evidence_items:
        if not isinstance(evidence, dict):
            raise ValueError(
                "Each retrieved evidence item must be a dictionary."
            )
        record_id = evidence.get("record_id")
        if (
            not isinstance(record_id, str)
            or not record_id
            or not record_id.strip()
            or record_id != record_id.strip()
        ):
            raise ValueError(
                f"Retrieved evidence item has an invalid record_id: "
                f"{record_id!r}."
            )

    for navigation_hit in navigation_hits:
        if not isinstance(navigation_hit, dict):
            raise ValueError(
                "Each retrieved navigation hit must be a dictionary."
            )

        document_id = navigation_hit.get("document_id")
        if (
            not isinstance(document_id, str)
            or not document_id
            or document_id != document_id.strip()
        ):
            raise ValueError(
                "Navigation hit document_id must be a non-empty string."
            )

        node_id = navigation_hit.get("node_id")
        if (
            not isinstance(node_id, str)
            or not node_id
            or node_id != node_id.strip()
        ):
            raise ValueError(
                "Navigation hit node_id must be a non-empty string."
            )

    if not evidence_items and not navigation_hits:
        return {
            "query": query,
            "retrieval": retrieval_result,
            "expanded_contexts": [],
            "evaluation": None,
            "loaded_documents": [],
        }

    # Resolve each document to its own structure file.
    document_registry = _build_document_registry(
        retrieval_result,
        structure_path,
        evidence_items + navigation_hits,
    )

    for evidence in evidence_items:
        document_id = evidence.get("document_id")
        if document_id is None:
            if "__legacy__" not in document_registry:
                raise ValueError(
                    "Retrieved evidence is missing document_id."
                )
            document_id = "__legacy__"

        if (
            not isinstance(document_id, str)
            or not document_id
            or document_id != document_id.strip()
        ):
            raise ValueError("Evidence document_id must be a non-empty string.")
        if document_id not in document_registry:
            raise ValueError(
                f"Retrieved evidence references an unregistered document_id: "
                f"{document_id!r}."
            )

    # Cache: document_id -> StructureIndex.
    # A document's structure is loaded only once per pipeline execution.
    structure_cache: dict[str, StructureIndex] = {}

    def get_structure_index(document_id: str) -> StructureIndex:
        if document_id not in document_registry:
            raise KeyError(
                f"Retrieved evidence references an unregistered document: "
                f"{document_id}"
            )

        if document_id not in structure_cache:
            path = document_registry[document_id]

            if not path.is_file():
                raise FileNotFoundError(
                    f"Structure file for document {document_id!r} "
                    f"was not found: {path}"
                )

            structure_index = StructureIndex.from_file(path)
            if structure_index.document_id != document_id:
                raise ValueError(
                    f"Structure file {path} contains document_id "
                    f"{structure_index.document_id!r}, but the requested "
                    f"document_id is {document_id!r}."
                )
            structure_cache[document_id] = structure_index

        return structure_cache[document_id]

    merged_evidence_items = []
    seen_record_ids = set()
    for evidence in evidence_items:
        record_id = evidence["record_id"]
        if record_id in seen_record_ids:
            continue
        seen_record_ids.add(record_id)
        merged_evidence_items.append(evidence)
    direct_evidence_count = len(merged_evidence_items)

    navigation_nodes_by_document: dict[str, list[str]] = {}
    for navigation_hit in navigation_hits:
        document_id = navigation_hit["document_id"]
        if document_id not in document_registry:
            raise ValueError(
                f"Navigation hit references an unregistered document_id: "
                f"{document_id!r}."
            )
        navigation_nodes_by_document.setdefault(document_id, []).append(
            navigation_hit["node_id"]
        )

    if navigation_nodes_by_document:
        from .retrieval import get_subtree_evidence_records

        for document_id, navigation_node_ids in navigation_nodes_by_document.items():
            structure_index = get_structure_index(document_id)
            subtree_node_ids: list[str] = []
            seen_node_ids: set[str] = set()

            for navigation_node_id in navigation_node_ids:
                for subtree_node_id in structure_index.get_subtree_node_ids(
                    navigation_node_id
                ):
                    if subtree_node_id not in seen_node_ids:
                        seen_node_ids.add(subtree_node_id)
                        subtree_node_ids.append(subtree_node_id)

            subtree_evidence = get_subtree_evidence_records(
                Path(db_path),
                document_id,
                subtree_node_ids,
            )

            for evidence in subtree_evidence:
                if not isinstance(evidence, dict):
                    raise ValueError(
                        "Subtree evidence lookup returned a non-dictionary "
                        "record."
                    )

                if evidence.get("document_id") != document_id:
                    raise ValueError(
                        "Subtree evidence lookup returned a record for the "
                        f"wrong document: expected {document_id!r}, got "
                        f"{evidence.get('document_id')!r}."
                    )

                record_id = evidence.get("record_id")
                if (
                    not isinstance(record_id, str)
                    or not record_id
                    or not record_id.strip()
                    or record_id != record_id.strip()
                ):
                    raise ValueError(
                        "Subtree evidence record has an invalid record_id: "
                        f"{record_id!r}."
                    )

                if record_id in seen_record_ids:
                    continue

                seen_record_ids.add(record_id)
                merged_evidence_items.append(evidence)

    evidence_items = merged_evidence_items

    for evidence in evidence_items[direct_evidence_count:]:
        document_id = evidence.get("document_id")
        if (
            not isinstance(document_id, str)
            or not document_id
            or document_id != document_id.strip()
        ):
            raise ValueError(
                "Subtree evidence document_id must be a non-empty string."
            )
        if document_id not in document_registry:
            raise ValueError(
                f"Subtree evidence references an unregistered document_id: "
                f"{document_id!r}."
            )

    stage = "context_expansion"
    stage_started = time.monotonic()
    expanded_contexts = []

    for evidence in evidence_items:
        # The legacy fallback is only for old single-document results
        # that do not contain document_id.
        document_id = evidence.get("document_id")

        if document_id is None:
            if "__legacy__" not in document_registry:
                raise ValueError(
                    "Retrieved evidence is missing document_id."
                )
            document_id = "__legacy__"

        if (
            not isinstance(document_id, str)
            or not document_id
            or document_id != document_id.strip()
        ):
            raise ValueError("Evidence document_id must be a non-empty string.")

        structure_index = get_structure_index(document_id)

        try:
            if evidence.get("node_id") == "__preamble__":
                expanded_context = _expand_preamble_context(
                    evidence, structure_index, before, after,
                )
            else:
                expanded_context = ContextExpander(
                    evidence, structure_index,
                ).expand_context(before=before, after=after)
        except Exception as exc:
            partial_result["expanded_contexts"] = expanded_contexts
            partial_result["loaded_documents"] = list(structure_cache.keys())
            raise PipelineStageError(stage, exc, partial_result) from exc

        expanded_context["document_id"] = evidence.get("document_id") or document_id
        expanded_context["record_id"] = evidence.get("record_id")
        expanded_contexts.append(expanded_context)

    partial_result["expanded_contexts"] = expanded_contexts
    partial_result["loaded_documents"] = list(structure_cache.keys())
    if progress_callback:
        progress_callback(stage, {"elapsed_seconds": time.monotonic() - stage_started,
                                  "retrieval": retrieval_result,
                                  "expanded_contexts": expanded_contexts,
                                  "loaded_documents": list(structure_cache.keys())})

    stage = "evaluation"
    stage_started = time.monotonic()
    try:
        evaluator = EvidenceEvaluator(query=query, expanded_contexts=expanded_contexts,
                                      llm_call=llm_call)
        evidence_count = evaluator.prepare_evidence()["total_unique_blocks"]
        if progress_callback:
            progress_callback(stage, {"evidence_blocks": evidence_count,
                                      "elapsed_seconds": 0.0,
                                      "retrieval": retrieval_result,
                                      "expanded_contexts": expanded_contexts,
                                      "loaded_documents": list(structure_cache.keys())})
        evaluation = evaluator.evaluate()
    except Exception as exc:
        raise PipelineStageError(stage, exc, partial_result) from exc
    if progress_callback:
        progress_callback(stage, {"evidence_blocks": evidence_count,
                                  "elapsed_seconds": time.monotonic() - stage_started,
                                  "retrieval": retrieval_result,
                                  "expanded_contexts": expanded_contexts,
                                  "loaded_documents": list(structure_cache.keys()),
                                  "completed": True})

    return {
        "query": query,
        "retrieval": retrieval_result,
        "expanded_contexts": expanded_contexts,
        "evaluation": evaluation,
        "loaded_documents": list(structure_cache.keys()),
    }
