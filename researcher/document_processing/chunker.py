from transformers import AutoTokenizer


class HierarchicalChunker:

    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=400,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens

    def count_tokens(self, text):
        return len(
            self.tokenizer.encode(
                text,
                add_special_tokens=False
            )
        )

    def split_large_paragraph(self, text, section_context):
        """
        Split a paragraph that exceeds the chunk token limit.
        Each resulting piece retains the section context.
        """
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False
        )

        context = f"Section: {section_context}\n\n"

        # Reserve tokens for the section heading.
        available_tokens = self.max_tokens - self.count_tokens(context)

        if available_tokens <= 0:
            raise ValueError(
                "Section heading is too long for the configured token limit."
            )

        pieces = []

        for start in range(0, len(token_ids), available_tokens):
            token_slice = token_ids[start:start + available_tokens]

            piece = self.tokenizer.decode(
                token_slice,
                skip_special_tokens=True
            )

            pieces.append(piece)

        return pieces

    def chunk_section(self, section, parent_path):
        heading = section.get("heading")
        level = section.get("level")

        current_path = parent_path.copy()

        if heading:
            current_path.append(heading)

        section_context = " > ".join(current_path)

        chunks = []
        current_items = []
        current_text = ""

        def build_chunk(items, body):
            if not items or not body.strip():
                return None

            page_numbers = sorted({
                page
                for item in items
                for page in item.get("page_numbers", [])
            })

            content_types = list(dict.fromkeys(
                item["type"] for item in items
            ))

            return {
                "chunk_id": None,
                "section_heading": heading,
                "section_level": level,
                "section_path": current_path.copy(),
                "text": body.strip(),
                "embedding_text": (
                    f"Section: {section_context}\n\n{body.strip()}"
                    if section_context else body.strip()
                ),
                "token_count": self.count_tokens(
                    (
                        f"Section: {section_context}\n\n"
                        if section_context else ""
                    ) + body.strip()
                ),
                "page_numbers": page_numbers,
                "content_types": content_types,
                "source_item_count": len(items),
            }

        for item in section.get("content", []):
            paragraph = item.get("text", "").strip()

            if not paragraph:
                continue

            item_copy = {
                "type": item.get("type", "unknown"),
                "text": paragraph,
                "page_numbers": item.get("page_numbers", []),
            }

            candidate_text = (
                f"{current_text}\n\n{paragraph}"
                if current_text else paragraph
            )

            context = (
                f"Section: {section_context}\n\n"
                if section_context else ""
            )

            candidate_tokens = self.count_tokens(
                context + candidate_text
            )

            # If adding this item exceeds the limit, save the current chunk.
            if current_items and candidate_tokens > self.max_tokens:
                chunk = build_chunk(current_items, current_text)
                if chunk:
                    chunks.append(chunk)

                current_items = []
                current_text = ""

            # Handle an individual paragraph larger than the token limit.
            single_item_tokens = self.count_tokens(
                context + paragraph
            )

            if single_item_tokens > self.max_tokens:
                pieces = self.split_large_paragraph(
                    paragraph,
                    section_context
                )

                for piece in pieces:
                    chunk = build_chunk([item_copy], piece)
                    if chunk:
                        chunks.append(chunk)

                continue

            if current_text:
                current_text += "\n\n" + paragraph
            else:
                current_text = paragraph

            current_items.append(item_copy)

        # Save the final chunk for this section.
        final_chunk = build_chunk(current_items, current_text)

        if final_chunk:
            chunks.append(final_chunk)

        # Process child sections independently.
        for child in section.get("children", []):
            chunks.extend(
                self.chunk_section(child, current_path)
            )

        return chunks

    def chunk_document(self, structure):
        chunks = self.chunk_section(structure, [])

        for index, chunk in enumerate(chunks, start=1):
            chunk["chunk_id"] = f"chunk_{index:04d}"

        return chunks