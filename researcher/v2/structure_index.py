
from collections import defaultdict
from pathlib import Path
import json


class StructureIndexError(Exception):
    """Raised when the document structure is invalid."""


class StructureIndex:
    def __init__(self, document: dict):
        self.document_id = document.get("document_id")
        self.doc_name = document.get("doc_name")
        self.doc_title = document.get("doc_title")

        # Lookup indexes
        self.nodes_by_id = {}
        self.parent_by_node_id = {}
        self.blocks_by_key = {}
        self.blocks_by_node_id = {}

        # Preamble is outside the normal node hierarchy.
        self.preamble = tuple(document.get("preamble", []))

        # Collect validation errors instead of failing at the
        # first invalid entry.
        self.errors = []

        self._build(document)
        self._validate()

        if self.errors:
            raise StructureIndexError(
                "Invalid document structure:\n"
                + "\n".join(f"- {error}" for error in self.errors)
            )

    @classmethod
    def from_file(cls, path: str | Path):
        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            document = json.load(file)

        return cls(document)

    def _build(self, document: dict):
        structure = document.get("structure")

        if not isinstance(structure, list):
            raise StructureIndexError(
                "'structure' must be a list."
            )

        for node in structure:
            self._visit_node(node, parent_id=None)

    def _visit_node(self, node: dict, parent_id: str | None):
        if not isinstance(node, dict):
            self.errors.append("Encountered a node that is not an object.")
            return

        node_id = node.get("node_id")

        if not isinstance(node_id, str) or not node_id:
            self.errors.append("Node is missing a valid node_id.")
            return

        if node_id in self.nodes_by_id:
            self.errors.append(f"Duplicate node_id: {node_id}")
            return

        self.nodes_by_id[node_id] = node
        self.parent_by_node_id[node_id] = parent_id

        content_blocks = node.get("content_blocks", [])

        if not isinstance(content_blocks, list):
            self.errors.append(
                f"Node {node_id}: content_blocks must be a list."
            )
            content_blocks = []

        # Preserve the original block ordering.
        ordered_blocks = []

        for block in content_blocks:
            if not isinstance(block, dict):
                self.errors.append(
                    f"Node {node_id}: encountered an invalid block."
                )
                continue

            block_id = block.get("block_id")

            if not isinstance(block_id, str) or not block_id:
                self.errors.append(
                    f"Node {node_id}: block is missing block_id."
                )
                continue

            block_key = (node_id, block_id)

            if block_key in self.blocks_by_key:
                self.errors.append(
                    f"Duplicate block ID in node {node_id}: {block_id}"
                )
                continue

            self.blocks_by_key[block_key] = block
            ordered_blocks.append(block)

        self.blocks_by_node_id[node_id] = tuple(ordered_blocks)

        children = node.get("children", [])

        if not isinstance(children, list):
            self.errors.append(
                f"Node {node_id}: children must be a list."
            )
            return

        for child in children:
            self._visit_node(child, parent_id=node_id)

    def _validate(self):
        # Validate preamble blocks.
        preamble_ids = set()

        for block in self.preamble:
            if not isinstance(block, dict):
                self.errors.append("Invalid preamble block.")
                continue

            block_id = block.get("block_id")

            if not isinstance(block_id, str) or not block_id:
                self.errors.append(
                    "Preamble block is missing block_id."
                )
                continue

            if block_id in preamble_ids:
                self.errors.append(
                    f"Duplicate preamble block ID: {block_id}"
                )

            preamble_ids.add(block_id)

        # Validate parent references.
        for node_id, parent_id in self.parent_by_node_id.items():
            if parent_id is not None and parent_id not in self.nodes_by_id:
                self.errors.append(
                    f"Node {node_id} references missing parent {parent_id}."
                )

    # -------------------------
    # Public lookup operations
    # -------------------------

    def get_node(self, node_id: str) -> dict:
        try:
            return self.nodes_by_id[node_id]
        except KeyError:
            raise KeyError(f"Node not found: {node_id}")

    def get_parent(self, node_id: str) -> dict | None:
        self.get_node(node_id)

        parent_id = self.parent_by_node_id[node_id]

        if parent_id is None:
            return None

        return self.get_node(parent_id)

    def get_blocks(self, node_id: str) -> tuple[dict, ...]:
        self.get_node(node_id)
        return self.blocks_by_node_id[node_id]

    def get_block(self, node_id: str, block_id: str) -> dict:
        key = (node_id, block_id)

        try:
            return self.blocks_by_key[key]
        except KeyError:
            raise KeyError(
                f"Block {block_id} not found in node {node_id}"
            )

    def get_block_position(self, node_id: str, block_id: str) -> int:
        blocks = self.get_blocks(node_id)

        for position, block in enumerate(blocks):
            if block["block_id"] == block_id:
                return position

        raise KeyError(
            f"Block {block_id} not found in node {node_id}"
        )

    def get_children(self, node_id: str) -> tuple[dict, ...]:
        node = self.get_node(node_id)
        return tuple(node.get("children", []))

    def get_summary(self, node_id: str) -> str | None:
        return self.get_node(node_id).get("summary")

    def stats(self) -> dict:
        return {
            "document_id": self.document_id,
            "total_nodes": len(self.nodes_by_id),
            "total_content_blocks": len(self.blocks_by_key),
            "preamble_blocks": len(self.preamble),
        }
