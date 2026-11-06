import json
# import chunks
# with open("storage/chunks.json", "r") as file:
#     chunks = json.load(file)

# text extractor from each chunk

# temp = {
#         "chunk_id": "chunk_0023",
#         "section_heading": "Mistral: 31.3 Applying the Nursing Process",
#         "section_level": 5,
#         "section_path": [
#             "6 Results and Analysis",
#             "6.2 Error Analysis with Anecdotal Examples",
#             "Mistral: 31.3 Applying the Nursing Process"
#         ],
#         "text": "BM25: 11.2 Sensory Impairments Basic Concepts\n\nDPR: 31.2 Basic Concepts\n\nDSI: 31.2 Basic Concepts\n\nSTAIR: 31.3 Applying the Nursing Process\n\nMistral: 9.3.5 Illiberal and hybrid regimes\n\nBM25: 15.3.1 Insurgencies/Civil Wars\n\nDPR: 15.2.2 External State-Sponsored Political Violence (State-Sponsored Terrorism)\n\nDSI: 15.2.1 Internal State-Sponsored Political Violence (Government Terrorism)\n\nSTAIR: 9.2.1 Institutional channels\n\nDSI, the error rate drops to 24.31%, with 3.25% of its predictions being non-leaf nodes (suggesting knowledge infusion helps). STAIR has the lowest error rate of 18.67%, with only 0.05% of its predictions being non-leaf nodes.\n\nTable 3 lists a few qualitative examples. We discuss next the first example \"How do preschoolers react when caregivers and teachers belittle their autonomous actions\" from the book \"The Whole Child: Development in the Early Years\":\n\nMistral could correctly guess that the query can be answered by \"2.2.3 Initiative vs. Guilt (Preschool Years)\" using its world knowledge. It doesn't understand that it needs to pick only one option - due to lack of the task training",
#         "embedding_text": "Section: 6 Results and Analysis > 6.2 Error Analysis with Anecdotal Examples > Mistral: 31.3 Applying the Nursing Process\n\nBM25: 11.2 Sensory Impairments Basic Concepts\n\nDPR: 31.2 Basic Concepts\n\nDSI: 31.2 Basic Concepts\n\nSTAIR: 31.3 Applying the Nursing Process\n\nMistral: 9.3.5 Illiberal and hybrid regimes\n\nBM25: 15.3.1 Insurgencies/Civil Wars\n\nDPR: 15.2.2 External State-Sponsored Political Violence (State-Sponsored Terrorism)\n\nDSI: 15.2.1 Internal State-Sponsored Political Violence (Government Terrorism)\n\nSTAIR: 9.2.1 Institutional channels\n\nDSI, the error rate drops to 24.31%, with 3.25% of its predictions being non-leaf nodes (suggesting knowledge infusion helps). STAIR has the lowest error rate of 18.67%, with only 0.05% of its predictions being non-leaf nodes.\n\nTable 3 lists a few qualitative examples. We discuss next the first example \"How do preschoolers react when caregivers and teachers belittle their autonomous actions\" from the book \"The Whole Child: Development in the Early Years\":\n\nMistral could correctly guess that the query can be answered by \"2.2.3 Initiative vs. Guilt (Preschool Years)\" using its world knowledge. It doesn't understand that it needs to pick only one option - due to lack of the task training",
#         "token_count": 306,
#         "page_numbers": [
#             7
#         ],
#         "content_types": [
#             "paragraph"
#         ],
#         "source_item_count": 12
#     }



def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def text_extractor(chunk):
    if not chunk or "embedding_text" not in chunk:
        return ""
    text = chunk["embedding_text"]
    
    return text

def extract_text_from_chunks(chunks):
    extracted_texts = []
    for chunk in chunks:
        text = text_extractor(chunk)
        extracted_texts.append(text)
    return extracted_texts


