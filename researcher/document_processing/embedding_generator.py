
import json
import numpy as np
import torch

from pathlib import Path
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:

    def __init__(
        self,
        model_name="Qwen/Qwen3-Embedding-0.6B",
        batch_size=8
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading embedding model on: {self.device}")

        self.model = SentenceTransformer(
            model_name,
            device=self.device
        )

        self.batch_size = batch_size

    def generate_embeddings(self, chunks):

        if not chunks:
            raise ValueError("No chunks provided.")

        chunk_ids = []
        embedding_texts = []

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            embedding_text = chunk.get("embedding_text")

            if not chunk_id:
                raise ValueError("A chunk is missing its chunk_id.")

            if not isinstance(embedding_text, str) or not embedding_text.strip():
                raise ValueError(
                    f"Invalid embedding_text in {chunk_id}"
                )

            chunk_ids.append(chunk_id)
            embedding_texts.append(embedding_text)

        embeddings = self.model.encode_document(
            embedding_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError("Chunk and embedding counts do not match.")

        return chunk_ids, embeddings

    def save_embeddings(
        self,
        chunks,
        chunk_ids,
        embeddings,
        output_dir
    ):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(
            output_dir / "embeddings.npy",
            embeddings
        )

        with open(
            output_dir / "embedded_chunks.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(chunks, file, ensure_ascii=False, indent=2)

        with open(
            output_dir / "embedding_metadata.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                {
                    "model": "Qwen/Qwen3-Embedding-0.6B",
                    "dimension": int(embeddings.shape[1]),
                    "total_chunks": len(chunk_ids),
                    "chunk_ids": chunk_ids
                },
                file,
                ensure_ascii=False,
                indent=2
            )

        print(f"Saved {len(embeddings)} embeddings.")
        print(f"Embedding dimensions: {embeddings.shape[1]}")
        print(f"Output directory: {output_dir}")
