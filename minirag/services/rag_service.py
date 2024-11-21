import ollama
import numpy as np
from typing import List

from minirag.models import Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter



class RagService:

    @staticmethod
    def get_splitter(chunk_size: int = 1000, chunk_overlap: int = 20) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def generate_embeddings(
            src_text: str,
            model_name: str = "all-minilm",
        ) -> List[float]:
        emb = ollama.embeddings(
            model=model_name,
            prompt=src_text,
        )
        return emb["embedding"]

    @staticmethod
    def similarity_search(
        query: str,
        collection: List[Chunk],
        top_k: int = 5,
    ) -> str:
        query_emb = RagService.generate_embeddings(query)
        for record in collection:
            record.similarity = np.dot(record.embedding, query_emb)
        collection.sort(key=lambda x: x.similarity, reverse=True)
        return " ".join([record.text for record in collection[:top_k]])
