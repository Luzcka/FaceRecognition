# core/config.py

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
import os
from pathlib import Path

class Settings(BaseSettings):
    """
    Configurações da aplicação carregadas a partir de variáveis de ambiente.
    """
    
    # API Configuration
    api_key: str = Field(default="supersecret", alias="API_KEY")
    secret_key: str = Field(default="dev-secret-key", alias="SECRET_KEY")
    
    # Milvus Configuration
    milvus_mode: str = Field(default="local", alias="MILVUS_MODE")
    milvus_host: str = Field(default="localhost", alias="MILVUS_HOST")
    milvus_port: int = Field(default=19530, alias="MILVUS_PORT")
    milvus_local_path: str = Field(default="data/milvus_faces.db", alias="MILVUS_LOCAL_PATH")
    
    # Face Recognition Configuration
    face_model: str = Field(default="Facenet512", alias="FACE_MODEL")
    face_detector: str = Field(default="opencv", alias="FACE_DETECTOR")
    similarity_threshold: float = Field(default=0.95, alias="SIMILARITY_THRESHOLD")
    top_k_results: int = Field(default=5, alias="TOP_K_RESULTS")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @property
    def is_local_mode(self) -> bool:
        """Verifica se está rodando em modo local."""
        return self.milvus_mode.lower() == "local"
    
    @property
    def embedding_dimension(self) -> int:
        """Retorna a dimensão do embedding baseado no modelo."""
        model_dimensions = {
            "Facenet": 128,
            "Facenet512": 512,
            "VGG-Face": 2622,
            "OpenFace": 128,
            "DeepFace": 4096,
            "ArcFace": 512,
            "SFace": 128
        }
        return model_dimensions.get(self.face_model, 512)

# Instância global das configurações
settings = Settings()