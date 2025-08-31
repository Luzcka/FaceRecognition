# models/db_info.py

from pydantic import BaseModel, Field
from typing import Optional



class CollectionInfo(BaseModel):
    """Informações da collection do Milvus."""
    collection_name: str
    dimension: int
    mode: str
    exists: bool
    total_records: int | str  # Pode ser "N/A"
    
class DatabaseRecord(BaseModel):
    """Registro individual da base."""
    id: Optional[int]
    name: str
    registration_number: str
    embedding_dimension: Optional[int] = None
    # embedding não incluído no modelo por ser muito grande
    
class SimilarityComparison(BaseModel):
    """Resultado de comparação de similaridade."""
    cosine_similarity: float
    cosine_score: float = Field(..., ge=0.0, le=1.0)
    euclidean_distance: float
    normalized_euclidean: float
    is_same_person: bool    