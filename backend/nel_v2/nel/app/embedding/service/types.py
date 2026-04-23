from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    text: str
    embedding: list[float]
    model_id: str
    dimensions: int
