from typing import List

from minirag.models import Chunk
from minirag.services.rag_service import RagService


class DocumentService:

    @staticmethod
    def read_document(doc_path: str) -> str:
        try:
            with open(doc_path, "r") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"File not found: {doc_path}")
            return ""
        except Exception as e:
            print(f"Error reading file: {doc_path}")
            print(e)
            return ""
        
        return text

    @staticmethod
    def process_document(doc_path: str) -> List[Chunk]:
        
        doc_text = DocumentService.read_document(doc_path)
        splitter = RagService.get_splitter()
        
        text_chunks = splitter.split_text(doc_text)

        chunks = []
        for chunk in text_chunks:
            emb = RagService.generate_embeddings(chunk)
            chunks.append(Chunk(doc_path, chunk, emb))

        return chunks