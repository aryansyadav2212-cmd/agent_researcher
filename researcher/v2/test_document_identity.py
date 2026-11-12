from __future__ import annotations

import copy
import re
import tempfile
import unittest
from pathlib import Path

from .document_identity import DocumentIdentityStore
from .markdown_extractor import extract_markdown_structure
from .navigation_summarizer import generate_navigation_summaries
from .structure_index import StructureIndex


UUID_ID = re.compile(r"^(doc|node|blk)_[0-9a-f]{32}$")


def _extract(markdown: str, *, identity: str, previous: dict | None = None) -> dict:
    return extract_markdown_structure(
        markdown,
        doc_name="sample.pdf",
        doc_title="Sample document",
        document_id=identity,
        previous_structure=previous,
    )


def _flatten_nodes(nodes: list[dict]) -> list[dict]:
    result = []
    for node in nodes:
        result.append(node)
        result.extend(_flatten_nodes(node.get("children", [])))
    return result


class PersistentDocumentIdentityTests(unittest.TestCase):
    def test_exact_pdf_reuses_persisted_document_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DocumentIdentityStore(temp_dir)
            first = store.resolve(b"%PDF-1.7 synthetic pdf")
            second = DocumentIdentityStore(temp_dir).resolve(b"%PDF-1.7 synthetic pdf")

        self.assertEqual(first.document_id, second.document_id)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertRegex(first.document_id, r"^[0-9a-f-]{36}$")

    def test_changed_pdf_gets_a_different_identity(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DocumentIdentityStore(temp_dir)
            first = store.resolve(b"%PDF-1.7 first synthetic pdf")
            second = store.resolve(b"%PDF-1.7 second synthetic pdf")

        self.assertNotEqual(first.document_id, second.document_id)

    def test_existing_unregistered_pdf_is_migrated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            legacy_dir = Path(temp_dir) / "legacy-document-id"
            legacy_dir.mkdir()
            (legacy_dir / "original.pdf").write_bytes(b"%PDF-1.4 legacy")

            identity = DocumentIdentityStore(temp_dir).resolve(b"%PDF-1.4 legacy")

        self.assertEqual(identity.document_id, "legacy-document-id")

    def test_reprocessing_preserves_ids_and_additions_get_new_ids(self):
        original_markdown = """## Alpha

Same paragraph.

Same paragraph.

## Beta

Unchanged beta content.
"""
        first = _extract(original_markdown, identity="doc_existing")
        previous_nodes = _flatten_nodes(first["structure"])
        previous_node_ids = {node["title"]: node["node_id"] for node in previous_nodes}
        previous_block_ids: dict[tuple[str, str], list[str]] = {}
        for node in previous_nodes:
            for block in node["content_blocks"]:
                previous_block_ids.setdefault((node["title"], block["text"]), []).append(
                    block["block_id"]
                )

        updated_markdown = """## Alpha

Same paragraph.

Inserted paragraph.

Same paragraph.

## Beta

Unchanged beta content.

## Gamma

New section content.
"""
        second = _extract(
            updated_markdown,
            identity="doc_existing",
            previous=first,
        )
        current_nodes = _flatten_nodes(second["structure"])
        current_by_title = {node["title"]: node for node in current_nodes}

        self.assertEqual(second["document_id"], first["document_id"])
        self.assertEqual(current_by_title["Alpha"]["node_id"], previous_node_ids["Alpha"])
        self.assertEqual(current_by_title["Beta"]["node_id"], previous_node_ids["Beta"])
        self.assertNotEqual(current_by_title["Gamma"]["node_id"], previous_node_ids["Alpha"])
        self.assertEqual(
            current_by_title["Beta"]["content_blocks"][0]["block_id"],
            previous_block_ids[("Beta", "Unchanged beta content.")][0],
        )
        alpha_text_ids = [
            block["block_id"]
            for block in current_by_title["Alpha"]["content_blocks"]
            if block["text"] == "Same paragraph."
        ]
        self.assertEqual(len(alpha_text_ids), 2)
        self.assertEqual(len(set(alpha_text_ids)), 2)
        self.assertEqual(
            set(alpha_text_ids),
            {
                block_id
                for block_id in previous_block_ids[("Alpha", "Same paragraph.")]
            },
        )
        inserted = next(
            block
            for block in current_by_title["Alpha"]["content_blocks"]
            if block["text"] == "Inserted paragraph."
        )
        self.assertNotIn(
            inserted["block_id"],
            {
                block_id
                for block_ids in previous_block_ids.values()
                for block_id in block_ids
            },
        )

    def test_duplicate_titles_are_matched_by_unique_content(self):
        markdown = """## Repeated

First section text.

## Repeated

Second section text.
"""
        first = _extract(markdown, identity="doc_duplicate_titles")
        first_ids = [node["node_id"] for node in first["structure"]]

        second = _extract(markdown, identity="doc_duplicate_titles", previous=first)

        self.assertEqual([node["node_id"] for node in second["structure"]], first_ids)
        self.assertEqual(len(set(first_ids)), 2)

    def test_different_pdfs_have_disjoint_document_node_and_block_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DocumentIdentityStore(temp_dir)
            first_identity = store.resolve(b"%PDF first")
            second_identity = store.resolve(b"%PDF second")
        markdown = "## Section\n\nContent block.\n"
        first = _extract(markdown, identity=first_identity.document_id)
        second = _extract(markdown, identity=second_identity.document_id)

        first_node_ids = {node["node_id"] for node in _flatten_nodes(first["structure"])}
        second_node_ids = {node["node_id"] for node in _flatten_nodes(second["structure"])}
        first_block_ids = {
            block["block_id"]
            for node in _flatten_nodes(first["structure"])
            for block in node["content_blocks"]
        }
        second_block_ids = {
            block["block_id"]
            for node in _flatten_nodes(second["structure"])
            for block in node["content_blocks"]
        }

        self.assertNotEqual(first["document_id"], second["document_id"])
        self.assertTrue(first_node_ids.isdisjoint(second_node_ids))
        self.assertTrue(first_block_ids.isdisjoint(second_block_ids))

    def test_summarizer_and_retrieval_consume_extracted_structure(self):
        tree = _extract(
            "## Section\n\nContent available for retrieval.\n",
            identity="doc_workflow",
        )
        navigation_result = generate_navigation_summaries(
            copy.deepcopy(tree),
            dry_run=True,
        )
        structure_index = StructureIndex(tree)

        self.assertEqual(
            navigation_result["summary_generation_plan"]["request_count"],
            1,
        )
        self.assertEqual(structure_index.document_id, tree["document_id"])
        self.assertEqual(structure_index.stats()["total_nodes"], 1)
        self.assertEqual(structure_index.stats()["total_content_blocks"], 1)


if __name__ == "__main__":
    unittest.main()
