# api/endpoints/tools.py

from fastapi import APIRouter, File, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any
from uuid import uuid4
import logging

from services.milvus_service import MilvusService
from models.db_info import CollectionInfo
from api.dependencies import get_milvus_service, validate_api_key

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/info", response_model=CollectionInfo)
async def get_collection_info(
    milvus_service: MilvusService = Depends(get_milvus_service),
    _: bool = Depends(validate_api_key)
):
    """
    Obtém informações detalhadas sobre a collection atual.
    Útil para monitoramento e debug da base de dados.
    
    Returns:
        Informações da collection incluindo nome, dimensão, total de registros, etc.
        
    Raises:
        HTTPException: Se houver erro ao obter informações
    """
    try:
        logger.info("Obtendo informações da collection")
        
        # Chama o método do serviço
        info_dict = milvus_service.get_collection_info()
        
        # Verifica se houve erro
        if "error" in info_dict:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao obter informações: {info_dict['error']}"
            )
        
        # Converte para o modelo Pydantic
        collection_info = CollectionInfo(**info_dict)
        
        logger.info(f"Informações obtidas: {collection_info.collection_name} - {collection_info.total_records} registros")
        
        return collection_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erro inesperado ao obter informações: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao obter informações da collection"
        )

@router.delete("/clear")
async def clear_database(
    confirm: bool = Query(
        False, 
        description="Deve ser True para confirmar a operação de limpeza. Esta ação é irreversível!"
    ),
    milvus_service: MilvusService = Depends(get_milvus_service),
    _: bool = Depends(validate_api_key)
):
    """
    Elimina completamente a collection e recria do zero.
    ---> OPERAÇÃO PERIGOSA - Remove todos os dados permanentemente! <---
    
    Args:
        confirm: Deve ser True para confirmar a operação
        
    Returns:
        Status da operação de limpeza
        
    Raises:
        HTTPException: Se confirmação não for fornecida ou houver erro
    """
    
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Confirmação necessária",
                "message": "Para limpar a base, você deve passar confirm=true como query parameter",
                "example": "/api/v1/tools/clear?confirm=true",
                "warning": "Esta operação é irreversível e remove TODOS os dados!"
            }
        )
    
    try:
        logger.warning("INICIANDO LIMPEZA COMPLETA DA BASE DE DADOS")
        
        # Obtém informações antes da limpeza para log
        try:
            info_before = milvus_service.get_collection_info()
            records_before = info_before.get("total_records", "N/A")
            logger.info(f"Registros antes da limpeza: {records_before}")
        except:
            records_before = "N/A"
        
        # Executa a limpeza
        success = milvus_service.clear_all_data(confirm=True)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Falha ao limpar a base de dados"
            )
        
        # Obtém informações após limpeza para confirmação
        try:
            info_after = milvus_service.get_collection_info()
            records_after = info_after.get("total_records", 0)
        except:
            records_after = 0
        
        logger.warning(f"BASE LIMPA COM SUCESSO! Registros removidos: {records_before}")
        
        return {
            "status": "success",
            "message": "Base de dados limpa e recriada com sucesso",
            "details": {
                "records_before": records_before,
                "records_after": records_after,
                "collection_name": info_after.get("collection_name", "N/A"),
                "dimension": info_after.get("dimension", "N/A")
            },
            "warning": "Todos os dados foram permanentemente removidos"
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        # Erro de validação do método clear_all_data
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Erro inesperado na limpeza: {e}")
        raise HTTPException(
            status_code=500,
            detail="Erro interno ao limpar base de dados"
        )

@router.get("/health")
async def tools_health():
    """
    Health check específico para os endpoints de tools.
    """
    return {
        "status": "healthy",
        "service": "tools",
        "endpoints": [
            "GET /info - Informações da collection",
            "DELETE /clear - Limpar base de dados",
            "GET /health - Health check"
        ]
    }
