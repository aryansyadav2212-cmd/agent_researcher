from docling_core.types.doc import DoclingDocument
import json
def load_saved_document(json_path: str):
    doc = DoclingDocument.load_from_json(json_path)
    return doc

from structure_extractor import extract_document_structure

# DOCLING_JSON = "storage/paper.json"

# # Load previously extracted document
# doc = load_saved_document(DOCLING_JSON)

# # Extract its heading hierarchy and content
# structure = extract_document_structure(doc)

# # print(json.dumps(structure, indent=2))
# with open("storage/structured_document.json", "w", encoding="utf-8") as f:
#     json.dump(structure, f, indent=4, ensure_ascii=False)















import json
from pathlib import Path

from chunker import HierarchicalChunker

STRUCTURE_JSON = Path(
    "storage/structured_document.json"
)

CHUNKS_JSON = Path(
    "storage/chunks.json"
)

# Load Stage 1 output
with open(STRUCTURE_JSON, "r", encoding="utf-8") as file:
    structure = json.load(file)

# Initialize chunker
chunker = HierarchicalChunker(
    max_tokens=400
)

# Stage 2
chunks = chunker.chunk_document(structure)

# Save chunks
with open(CHUNKS_JSON, "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=4)

print(f"Generated {len(chunks)} chunks.")
print(f"Saved chunks to: {CHUNKS_JSON}")