import os
import numpy as np
from typing import List, Optional

from models import Chunk
from services.document_service import DocumentService


class CollectionService:
    def __init__(
        self,
        storage_path: str = "collections",
    ) -> None:
        self.storage_path = storage_path
        self.active_collection: Optional[List[Chunk]] = None

    def _process_folder(self, folder_path: str) -> List[Chunk]:
        doc_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]
        chunks = []
        for doc_path in doc_paths:
            doc_chunks = DocumentService.process_document(doc_path)
            chunks.extend(doc_chunks)
        return chunks
    
    def _store_embeddings(
        self,
        doc_chunks: List[Chunk],
        collection_name: str,
    ) -> None:
        
        records_np = np.array(doc_chunks)
        
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)
        np.save(f"{self.storage_path}/{collection_name}.npy", records_np)

    def create_collection(
            self,
        doc_paths: List[str],
        collection_name: str,
    ) -> None:
        print(f"Creating collection: {collection_name}...")
        records: List[Chunk] = []

        for doc_path in doc_paths:
            if os.path.isdir(doc_path):
                doc_chunks = self._process_folder(doc_path)
            else:
                doc_chunks = DocumentService.process_document(doc_path)
            records.extend(doc_chunks)

        self._store_embeddings(records, collection_name)
        print(f"Collection {collection_name} created")


    def load_collection(self, collection_name: str) -> None:
        try:
            collection_path = f"{self.storage_path}/{collection_name}.npy"
            print(collection_path)
            records = np.load(collection_path, allow_pickle=True)
            print(f"Loading collection: {collection_name}..., {records.shape[0]} records")
            self.active_collection = records.tolist()
            print(f"Collection {collection_name} loaded")
        except FileNotFoundError:
            print(f"Collection {collection_name} not found")
        
    def list_collections(self) -> None:
        collections = [f.split(".")[0] for f in os.listdir(self.storage_path)]
        
        print("Available collections:")
        print(collections)