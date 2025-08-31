# api/dependencies.py

from fastapi import Depends, HTTPException
from core.security import security_service
from services.face_service import FaceService
from services.milvus_service import milvus_service

def get_face_service() -> FaceService:
    """Dependency para obter instância do FaceService."""
    return FaceService()

def get_milvus_service():
    """Dependency para obter instância do MilvusService."""
    return milvus_service

def validate_api_key():
    """Dependency para validação da API key."""
    return Depends(security_service.validate_api_key)