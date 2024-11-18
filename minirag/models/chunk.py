from dataclasses import dataclass


@dataclass
class Chunk:
    document_name: str
    text: str
    embedding: list[float]
    