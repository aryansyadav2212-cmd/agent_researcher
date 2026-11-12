"""Document selection using the existing hybrid retrieval index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .retrieval import list_indexed_documents, search_index


class InvalidDocumentSelection(ValueError):
    """Raised when a caller explicitly targets unregistered documents."""


def route_query(
    query: str,
    db_path: str | Path,
    *,
    document_ids: list[str] | None = None,
    top_k: int = 8,
    candidate_k: int = 30,
) -> dict[str, Any]:
    """Select indexed documents explicitly or from corpus-wide hybrid hits."""
    registry = list_indexed_documents(Path(db_path))
    registry_by_id = {
        document["document_id"]: document
        for document in registry
    }

    if document_ids is not None:
        if not document_ids:
            raise InvalidDocumentSelection("At least one document_id is required.")
        if any(
            not isinstance(document_id, str)
            or not document_id
            or document_id != document_id.strip()
            for document_id in document_ids
        ):
            raise InvalidDocumentSelection(
                "Every document_id must be a non-empty, trimmed string."
            )
        if len(set(document_ids)) != len(document_ids):
            raise InvalidDocumentSelection("document_ids must not contain duplicates.")
        invalid = [document_id for document_id in document_ids if document_id not in registry_by_id]
        if invalid:
            raise InvalidDocumentSelection(
                "Unknown or invalid document_id(s): "
                + ", ".join(str(document_id) for document_id in invalid)
            )
        return {
            "mode": "explicit",
            "selected_document_ids": list(document_ids),
            "document_scores": {},
            "registry": registry,
            "routing_retrieval": None,
        }

    routing_retrieval = search_index(
        query,
        Path(db_path),
        top_k=top_k,
        candidate_k=candidate_k,
    )
    hits = routing_retrieval.get("evidence", []) + routing_retrieval.get(
        "navigation_hits", []
    )

    scores: dict[str, float] = {}
    for hit in hits:
        document_id = hit.get("document_id")
        if document_id not in registry_by_id:
            continue
        score = float(hit.get("fusion_score") or 0.0)
        scores[document_id] = max(scores.get(document_id, float("-inf")), score)

    selected = sorted(scores, key=lambda document_id: (-scores[document_id], document_id))
    return {
        "mode": "automatic",
        "selected_document_ids": selected,
        "document_scores": scores,
        "registry": registry,
        "routing_retrieval": routing_retrieval,
    }
