# api/endpoints/register.py

from fastapi import APIRouter, UploadFile, Form, Depends, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from typing import Dict, Any
import tempfile
import aiofiles
from uuid import uuid4
import logging

from services.face_service import FaceService
from services.milvus_service import MilvusService
from models.user import User
from api.dependencies import get_face_service, get_milvus_service, validate_api_key

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/register", response_model=Dict[str, Any])
async def register_user(
    name: str = Form(..., description="Nome completo do usuário"),
    registration_number: str = Form(..., description="Número de registro único"),
    image: UploadFile = File(..., description="Imagem do rosto do usuário"),
    face_service: FaceService = Depends(get_face_service),
    milvus_service: MilvusService = Depends(get_milvus_service),
    _: bool = Depends(validate_api_key)
):
    """
    Registra um novo usuário no sistema de reconhecimento facial.
    
    Args:
        name: Nome completo do usuário
        registration_number: Número de registro único
        image: Arquivo de imagem contendo o rosto
        
    Returns:
        Status do registro
        
    Raises:
        HTTPException: Se houver erro no processamento
    """
    
    # Valida dados do usuário
    try:
        user_data = User(name=name, registration_number=registration_number)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Dados inválidos: {e}")
    
    # Valida tipo de arquivo
    if not image.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Arquivo deve ser uma imagem"
        )
    
    # Cria arquivo temporário
    temp_dir = Path(tempfile.gettempdir())
    temp_filename = f"face_{uuid4()}.jpg"
    temp_path = temp_dir / temp_filename
    
    try:
        # Salva imagem temporariamente
        async with aiofiles.open(temp_path, "wb") as out_file:
            content = await image.read()
            await out_file.write(content)
        
        logger.info(f"Processando registro para: {user_data.name}")
        
        # Extrai embedding
        embedding = face_service.extract_embedding(temp_path)
        
        if embedding is None:
            raise HTTPException(
                status_code=400,
                detail="Não foi possível detectar rosto na imagem. Verifique se a imagem contém um rosto visível."
            )
        
        # Insere no Milvus
        success = milvus_service.insert_embedding(
            embedding=embedding,
            name=user_data.name,
            registration_number=user_data.registration_number
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Erro ao salvar dados no banco de vetores"
            )
        
        logger.info(f"Usuário registrado com sucesso: {user_data.name}")
        
        return {
            "status": "success",
            "message": "Usuário registrado com sucesso",
            "user": {
                "name": user_data.name,
                "registration_number": user_data.registration_number
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro inesperado no registro: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erro interno do servidor"
        )
    
    finally:
        # Remove arquivo temporário
        if temp_path.exists():
            temp_path.unlink()