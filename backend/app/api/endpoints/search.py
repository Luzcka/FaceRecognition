# api/endpoints/search.py

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import List, Dict, Any
import tempfile
import aiofiles
from uuid import uuid4
import logging

from services.face_service import FaceService
from services.milvus_service import MilvusService
from models.user import UserSearchResult
from api.dependencies import get_face_service, get_milvus_service, validate_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/search", response_model=List[UserSearchResult])
async def search_user(
    image: UploadFile = File(..., description="Imagem para busca facial"),
    top_k: int = Query(default=5, ge=1, le=20, description="Número máximo de resultados"),
    face_service: FaceService = Depends(get_face_service),
    milvus_service: MilvusService = Depends(get_milvus_service),
    _: bool = Depends(validate_api_key)
):
    """
    Busca usuários similares baseado em reconhecimento facial.
    
    Args:
        image: Arquivo de imagem para busca
        top_k: Número máximo de resultados a retornar
        
    Returns:
        Lista de usuários encontrados ordenados por similaridade
        
    Raises:
        HTTPException: Se houver erro no processamento
    """
    
    # Valida tipo de arquivo
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="Arquivo deve ser uma imagem"
        )
    
    # Cria arquivo temporário
    temp_dir = Path(tempfile.gettempdir())
    temp_filename = f"search_{uuid4()}.jpg"
    temp_path = temp_dir / temp_filename
    
    try:
        # Salva imagem temporariamente
        async with aiofiles.open(temp_path, "wb") as out_file:
            content = await image.read()
            await out_file.write(content)
        
        logger.info("Processando busca facial")
        
        # Extrai embedding
        query_embedding = face_service.extract_embedding(temp_path)
        
        if query_embedding is None:
            raise HTTPException(
                status_code=400,
                detail="Não foi possível detectar rosto na imagem. Verifique se a imagem contém um rosto visível."
            )
        
        print(milvus_service.collection.describe())
        # Busca no Milvus
        results = milvus_service.search_similar_embeddings(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        logger.info(f"Encontrados {len(results)} usuários similares")
        
        # Ordena por similaridade (maior para menor)
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro inesperado na busca: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erro interno do servidor"
        )
    
    finally:
        # Remove arquivo temporário
        if temp_path.exists():
            temp_path.unlink()

@router.get("/search/health")
async def search_health():
    """Endpoint de health check para o serviço de busca."""
    return {"status": "healthy", "service": "search"}