from docling.document_converter import DocumentConverter

from docling.chunking import HybridChunker

from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer
)

from transformers import AutoTokenizer



from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    HeadingHierarchyOptions,
)


pipeline_options = PdfPipelineOptions()

pipeline_options.heading_hierarchy_options = (
    HeadingHierarchyOptions(enabled=True)
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    }
)

result = converter.convert("https://arxiv.org/pdf/2609.03874")
doc = result.document
# First run: convert and save

doc.save_as_json("storage/paper.json")
# markdown_output = document.export_to_markdown()
# print(markdown_output)

# print(document.export_to_markdown()[:10000])
# from docling_core.types.doc import DocItemLabel

# for item in doc.texts:
#     if item.label == DocItemLabel.SECTION_HEADER:
#         print(f"Level: {item.level} | {item.text}")

# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# MAX_TOKENS = 256

# # Load the embedding model's tokenizer
# hf_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

# tokenizer = HuggingFaceTokenizer(
#     tokenizer=hf_tokenizer,
#     max_tokens=MAX_TOKENS
# )

# # Configure chunker
# chunker = HybridChunker(
#     tokenizer=tokenizer,
#     merge_peers=True
# )

# chunks = list(chunker.chunk(dl_doc=document))

# # print(chunks[:2])

# for i, chunk in enumerate(chunks):

#     print(f"\n{'=' * 60}")
#     print(f"CHUNK {i}")

#     print("\nHEADINGS:")
#     print(chunk.meta.headings)

#     print("\nCONTENT:")
#     print(chunk.text)

#     print("\nCONTEXTUALIZED CONTENT:")
#     print(chunker.contextualize(chunk=chunk))





# from docling.document_converter import (
#     DocumentConverter,
#     PdfFormatOption,
# )
# from docling.datamodel.base_models import InputFormat
# from docling.datamodel.pipeline_options import (
#     PdfPipelineOptions,
#     HeadingHierarchyOptions,
# )

# pipeline_options = PdfPipelineOptions()

# pipeline_options.heading_hierarchy_options = (
#     HeadingHierarchyOptions(enabled=True)
# )

# converter = DocumentConverter(
#     format_options={
#         InputFormat.PDF: PdfFormatOption(
#             pipeline_options=pipeline_options
#         )
#     }
# )

# result = converter.convert("https://arxiv.org/pdf/2609.03874")
# doc = result.document




















