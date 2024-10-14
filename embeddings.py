from typing import List

import ollama
import os
import numpy as np


def create_collection(
    doc_paths: List[str],
    collection_name: str,
) -> None:
    print(f"Creating collection: {collection_name}...")
    embeddings: List[float] = []

    for doc_path in doc_paths:
        with open(doc_path, "r") as f:
            text = f.read()
            emb = _generate_embeddings(text)
            embeddings.append(emb)

    embeddings = np.array(embeddings)
    
    if not os.path.exists("collections"):
        os.makedirs("collections")
    np.save(f"collections/{collection_name}.npy", embeddings)


def _generate_embeddings(
        src_text: str,
        model_name: str = "all-minilm",
    ) -> List[float]:
    emb = ollama.embeddings(
        model=model_name,
        prompt=src_text,
    )
    return emb["embedding"]
