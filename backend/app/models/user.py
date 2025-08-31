# models/user.py

from pydantic import BaseModel, Field, validator
from typing import Optional
import re

class User(BaseModel):
    """
    Modelo de usuário para registro no sistema.
    """
    name: str = Field(..., min_length=2, max_length=100, description="Nome completo do usuário")
    registration_number: str = Field(..., min_length=3, max_length=50, description="Número de registro único")
    
    @validator('name')
    def validate_name(cls, v):
        """Valida o nome do usuário."""
        if not v.strip():
            raise ValueError('Nome não pode ser vazio')
        return v.strip()
    
    @validator('registration_number')
    def validate_registration_number(cls, v):
        """Valida o número de registro."""
        if not re.match(r'^[A-Za-z0-9\-_]+$', v):
            raise ValueError('Número de registro deve conter apenas letras, números, hífen e underscore')
        return v.upper()

class UserSearchResult(BaseModel):
    """
    Resultado da busca de usuário.
    """
    name: str
    registration_number: str
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Score de similaridade (0-1)")
    distance: float = Field(..., ge=0.0, description="Distância euclidiana")