from typing import List

import ollama
import os
import numpy as np
from models import Chunk


def create_collection(
    doc_paths: List[str],
    collection_name: str,
) -> None:
    print(f"Creating collection: {collection_name}...")
    records: List[Chunk] = []

    for doc_path in doc_paths:
        with open(doc_path, "r") as f:
            text = f.read()
            emb = _generate_embeddings(text)
            records.append(Chunk(doc_path, text, emb))

    records_np = np.array(records)
    
    if not os.path.exists("collections"):
        os.makedirs("collections")
    np.save(f"collections/{collection_name}.npy", records_np)


def _generate_embeddings(
        src_text: str,
        model_name: str = "all-minilm",
    ) -> List[float]:
    emb = ollama.embeddings(
        model=model_name,
        prompt=src_text,
    )
    return emb["embedding"]
