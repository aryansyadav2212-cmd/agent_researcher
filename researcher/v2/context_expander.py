
class ContextExpander:
    def __init__(self, retrieval_output: dict, structure_index):
        self.retrieval_output = retrieval_output
        self.structure_index = structure_index

        self.node_id = retrieval_output.get("node_id")
        self.block_id = retrieval_output.get("content_block_id")

        if not self.node_id or not self.block_id:
            raise ValueError(
                "Retrieval output must contain node_id "
                "and content_block_id."
            )

    def expand_context(self, before=2, after=2):

        # 1. Validate expansion boundaries.
        if not isinstance(before, int) or isinstance(before, bool) or before < 0:
            raise ValueError("'before' must be a non-negative integer.")

        if not isinstance(after, int) or isinstance(after, bool) or after < 0:
            raise ValueError("'after' must be a non-negative integer.")

        # 2. Validate node and retrieve anchor block.
        self.structure_index.get_node(self.node_id)

        anchor_block = self.structure_index.get_block(
            self.node_id,
            self.block_id
        )

        # 3. Retrieve all ordered blocks from the node.
        neighboring_blocks = self.structure_index.get_blocks(
            self.node_id
        )

        # 4. Find anchor position.
        anchor_position = self.structure_index.get_block_position(
            self.node_id,
            self.block_id
        )

        # 5. Calculate context boundaries.
        start_idx = max(0, anchor_position - before)
        end_idx = min(
            len(neighboring_blocks),
            anchor_position + after + 1
        )

        # 6. Retrieve context window.
        context_blocks = neighboring_blocks[start_idx:end_idx]

        # 7. Add position and anchor metadata.
        formatted_blocks = []

        for position in range(start_idx, end_idx):
            block = neighboring_blocks[position]

            formatted_blocks.append({
                **block,
                "position": position,
                "is_anchor": block["block_id"] == self.block_id
            })

        # 8. Return structured context.
        return {
            "node_id": self.node_id,
            "anchor_block_id": self.block_id,
            "anchor_position": anchor_position,
            "window": {
                "start_position": start_idx,
                "end_position": end_idx - 1
            },
            "blocks": formatted_blocks
        }
