import json
from pathlib import Path

from document_processing.document_loader import load_saved_document
from document_processing.structure_extractor import (
    extract_document_structure,
)

# PAPER_DIR = Path("storage/papers/paper_001")

DOCLING_JSON = "document_processing/storage/paper.json"
STRUCTURE_JSON = "document_processing/storage/structured_document.json"


# Step 1: Load previously converted document
doc = load_saved_document(DOCLING_JSON)

# Step 2: Reconstruct document hierarchy
structure = extract_document_structure(doc)

# Step 3: Save reconstructed structure
with open(STRUCTURE_JSON, "w", encoding="utf-8") as file:
    json.dump(
        structure,
        file,
        ensure_ascii=False,
        indent=2,
    )

print("Document structure extracted successfully!")
print(f"Saved at: {STRUCTURE_JSON}")