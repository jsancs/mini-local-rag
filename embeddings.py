from typing import List

import ollama
import os
import numpy as np
from models import Chunk
from utils import read_document
from langchain_text_splitters import RecursiveCharacterTextSplitter


COLLECTIONS_DIR = "collections"


def _get_splitter(chunk_size: int = 1000, chunk_overlap: int = 20) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

def _process_document(
    doc_path: str,
) -> List[Chunk]:
    
    doc_text = read_document(doc_path)
    splitter = _get_splitter()
    
    text_chunks = splitter.split_text(doc_text)

    chunks = []
    for chunk in text_chunks:
        emb = _generate_embeddings(chunk)
        chunks.append(Chunk(doc_path, chunk, emb))

    return chunks


def _process_folder(
    folder_path: str,
) -> List[Chunk]:
    doc_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]
    chunks = []
    for doc_path in doc_paths:
        doc_chunks = _process_document(doc_path)
        chunks.extend(doc_chunks)
    return chunks


def _generate_embeddings(
        src_text: str,
        model_name: str = "all-minilm",
    ) -> List[float]:
    emb = ollama.embeddings(
        model=model_name,
        prompt=src_text,
    )
    return emb["embedding"]


def _save_embeddings(
    doc_chunks: List[Chunk],
    collection_name: str,
) -> None:
    
    records_np = np.array(doc_chunks)
    
    if not os.path.exists(COLLECTIONS_DIR):
        os.makedirs(COLLECTIONS_DIR)
    np.save(f"{COLLECTIONS_DIR}/{collection_name}.npy", records_np)
    
    
def similarity_search(
    query: str,
    collection: List[Chunk],
    top_k: int = 5,
) -> str:
    query_emb = _generate_embeddings(query)
    for record in collection:
        record.similarity = np.dot(record.embedding, query_emb)
    collection.sort(key=lambda x: x.similarity, reverse=True)
    return " ".join([record.text for record in collection[:top_k]])


def create_collection(
    doc_paths: List[str],
    collection_name: str,
) -> None:
    print(f"Creating collection: {collection_name}...")
    records: List[Chunk] = []

    for doc_path in doc_paths:
        if os.path.isdir(doc_path):
            doc_chunks = _process_folder(doc_path)
        else:
            doc_chunks = _process_document(doc_path)
        records.extend(doc_chunks)

    _save_embeddings(records, collection_name)
    print(f"Collection {collection_name} created")


def load_collection(
    collection_name: str,
) -> List[Chunk]:
    try:
        records = np.load(f"{COLLECTIONS_DIR}/{collection_name}.npy", allow_pickle=True)
        return records.tolist()
    except FileNotFoundError:
        print(f"Collection {collection_name} not found")
        return []
