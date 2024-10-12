from typing import List

import ollama


def generate_embeddings(src_text: str, model_name: str) -> List[float]:
    emb = ollama.embeddings(
        model=model_name,
        prompt=src_text,
    )
    return emb