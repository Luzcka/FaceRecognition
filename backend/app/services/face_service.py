# services/face_service.py

from deepface import DeepFace
from pathlib import Path
import numpy as np
import cv2
from typing import Optional, Tuple
import logging
from core.config import settings

logger = logging.getLogger(__name__)

class FaceService:
    """
    Serviço para extração e comparação de embeddings faciais utilizando DeepFace.
    """
    
    def __init__(self, model_name: str = None, detector_backend: str = None):
        """
        Inicializa o serviço de reconhecimento facial.
        
        Args:
            model_name: Nome do modelo de reconhecimento facial
            detector_backend: Backend do detector de faces
        """
        self.model_name = model_name or settings.face_model
        self.detector_backend = detector_backend or settings.face_detector
        self.embedding_dimension = settings.embedding_dimension
        
        logger.info(f"Inicialized FaceService com modelo: {self.model_name}")
        logger.info(f"Dimensão do embedding: {self.embedding_dimension}")
    
    def extract_embedding(self, image_path: Path) -> Optional[np.ndarray]:
        """
        Extrai o vetor de embedding normalizado da imagem fornecida.
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Embedding normalizado ou None se houver erro
        """
        try:
            # Verifica se o arquivo existe
            if not image_path.exists():
                logger.error(f"Arquivo não encontrado: {image_path}")
                return None
            
            # Extrai o embedding usando DeepFace
            result = DeepFace.represent(
                img_path=str(image_path),
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=True,
                align=True
            )
            
            # DeepFace retorna uma lista de resultados
            if not result:
                logger.error("Nenhuma face detectada na imagem")
                return None
            
            # Pega o primeiro resultado (face principal)
            embedding = np.array(result[0]['embedding'], dtype=np.float32)
            
            # Normaliza o embedding para melhor comparação
            # Linha desnecessaria e errada para usar DeepFace, ele já normaliza.
            # normalized_embedding = self._normalize_embedding(embedding)
            
            logger.info(f"Embedding extraído com sucesso. Dimensão: {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Erro ao extrair embedding: {e}")
            return None
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normaliza o embedding usando L2 norm.
        
        Args:
            embedding: Vetor de embedding
            
        Returns:
            Embedding normalizado
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[float, float]:
        """
        Calcula a similaridade entre dois embeddings.
        
        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding
            
        Returns:
            Tupla com (distância euclidiana, score de similaridade)
        """
        # Calcula distância Cosseno
        cosine_distance = 1.0 - np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

        # Calcula similaridade coseno
        similarity_score = 1.0 - cosine_distance
        
        return cosine_distance, similarity_score

    def is_same_person(self, embedding1: np.ndarray, embedding2: np.ndarray) -> bool:
        """
        Determina se dois embeddings representam a mesma pessoa.
        
        Args:
            embedding1: Primeiro embedding
            embedding2: Segundo embedding
            
        Returns:
            True se for a mesma pessoa
        """
        _, similarity = self.calculate_similarity(embedding1, embedding2)
        return similarity >= settings.similarity_threshold