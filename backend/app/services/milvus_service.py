# services/milvus_service.py

from pymilvus import (
    connections, Collection, CollectionSchema,
    FieldSchema, DataType, utility, MilvusClient, __version__
)
from typing import List, Dict, Optional
import numpy as np
import logging
from core.config import settings
from models.user import UserSearchResult

logger = logging.getLogger(__name__)

class MilvusService:
    """
    Serviço para gerenciamento de vetores no Milvus.
    Compatível com diferentes versões e modos (local/remoto).
    """
    
    def __init__(self):
        """Inicializa o serviço do Milvus."""
        self.collection_name = f"face_embeddings_{settings.face_model}"
        self.dimension = settings.embedding_dimension
        self.client = None
        self.collection = None
        self.milvus_version = __version__
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Inicializa a conexão com o Milvus."""
        try:
            logger.info(f"Inicializando Milvus versão: {self.milvus_version}")
            
            if settings.is_local_mode:
                self._setup_local_milvus()
            else:
                self._setup_remote_milvus()
            
            logger.info(f"Milvus inicializado em modo {settings.milvus_mode}")
            
        except Exception as e:
            logger.error(f"Erro ao inicializar Milvus: {e}")
            raise
    
    def _setup_local_milvus(self):
        """Configura Milvus em modo local."""
        from pathlib import Path
        
        # Cria diretório se não existir
        db_path = Path(settings.milvus_local_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.client = MilvusClient(str(db_path))
        
        # Cria collection se não existir
        if not self.client.has_collection(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.dimension,
                primary_field="id",
                id_type="int64",
                metric_type="COSINE",  
                auto_id=True,
                description="Face embeddings collection"
            )
            logger.info(f"Collection '{self.collection_name}' criada (LOCAL) com métrica COSINE")
    
    def _setup_remote_milvus(self):
        """Configura Milvus em modo remoto."""
        connections.connect(
            "default",
            host=settings.milvus_host,
            port=settings.milvus_port
        )
        
        # Define schema da collection
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="registration_number", dtype=DataType.VARCHAR, max_length=50)
        ]
        
        schema = CollectionSchema(fields, description="Face Embeddings Collection")
        
        # Cria collection se não existir
        if not utility.has_collection(self.collection_name):
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Índice otimizado para COSINE
            index_params = {
                "metric_type": "COSINE",  # Embeddings RAW com métrica COSINE
                "index_type": "HNSW",
                "params": {
                    "M": 16,
                    "efConstruction": 200
                }
            }
            
            self.collection.create_index(field_name="embedding", index_params=index_params)
            logger.info(f"Collection '{self.collection_name}' criada (REMOTO) com métrica COSINE")
        else:
            self.collection = Collection(name=self.collection_name)
        
        # Carrega collection na memória
        self.collection.load()
        entity_count = self.collection.num_entities
        logger.debug(f"Collection {self.collection_name} carregada com {entity_count} registros")
   
    def insert_embedding(self, embedding: np.ndarray, name: str, registration_number: str) -> bool:
        """
        Insere um embedding RAW (não normalizado) no Milvus.
        
        Args:
            embedding: Vetor de embedding RAW (como extraído pelo DeepFace)
            name: Nome do usuário
            registration_number: Número de registro
            
        Returns:
            True se inserido com sucesso
        """
        try:
            # Usar embedding RAW (sem normalização)
            # O Milvus com métrica COSINE fará a normalização internamente
            raw_embedding = embedding.astype(np.float32)
            
            logger.debug(f"Inserindo embedding RAW - Norma: {np.linalg.norm(raw_embedding):.6f}")
            
            data = {
                "embedding": [raw_embedding.tolist()],
                "name": [name],
                "registration_number": [registration_number]
            }
            
            if settings.is_local_mode:
                self.client.insert(self.collection_name, data)
            else:
                self.collection.insert([
                    data["embedding"],
                    data["name"],
                    data["registration_number"]
                ])
            
            logger.info(f"Embedding inserido: {name} ({registration_number})")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao inserir embedding: {e}")
            return False
    
    def search_similar_embeddings(self, query_embedding: np.ndarray, top_k: int = None) -> List[UserSearchResult]:
        """
        Busca embeddings similares no Milvus usando embedding RAW.
        
        Args:
            query_embedding: Embedding de consulta RAW (não normalizado)
            top_k: Número máximo de resultados
            
        Returns:
            Lista de resultados encontrados
        """
        top_k = top_k or settings.top_k_results
        
        try:
            # Usar embedding RAW na busca
            raw_query = query_embedding.astype(np.float32)
            
            logger.debug(f"Buscando com embedding RAW - Norma: {np.linalg.norm(raw_query):.6f}")
            
            if settings.is_local_mode:
                results = self.client.search(
                    collection_name=self.collection_name,
                    data=[raw_query.tolist()],
                    output_fields=["name", "registration_number"],
                    limit=top_k,
                    search_params={"metric_type": "COSINE"}
                )
                
                return self._process_local_results(results)
            
            else:
                results = self.collection.search(
                    data=[raw_query.tolist()],
                    anns_field="embedding",
                    param={
                        "metric_type": "COSINE",
                        "params": {"ef": 200}
                    },
                    limit=top_k,
                    output_fields=["name", "registration_number"]
                )
                
                return self._process_remote_results(results)
                
        except Exception as e:
            logger.error(f"Erro na busca: {e}")
            return []
    
    def _process_local_results(self, results) -> List[UserSearchResult]:
        """
        Processa resultados do Milvus local (MilvusClient).
        
        IMPORTANTE: MilvusClient retorna 'score' (similaridade) em vez de 'distance'
        """
        processed_results = []
        
        if not results or not results[0]:
            return processed_results
        
        for hit in results[0]:
            try:
                # MilvusClient retorna 'score' (similaridade [0,1])
                similarity_score = hit.get("score", 0.0)
                
                # Converte para distância coseno [0,2]
                # similarity 1.0 -> distance 0.0 (idênticos)
                # similarity 0.0 -> distance 2.0 (opostos)
                cosine_distance = 2.0 * (1.0 - similarity_score)
                
                logger.debug(f"LOCAL - Score: {similarity_score:.6f}, Distance: {cosine_distance:.6f}")
                
                # Filtra por threshold
                if similarity_score >= settings.similarity_threshold:
                    processed_results.append(UserSearchResult(
                        name=hit["entity"]["name"],
                        registration_number=hit["entity"]["registration_number"],
                        similarity_score=similarity_score,
                        distance=cosine_distance
                    ))
                    
            except Exception as e:
                logger.error(f"Erro ao processar resultado local: {e}")
        
        return processed_results
    
    def _process_remote_results(self, results) -> List[UserSearchResult]:
        """
        Processa resultados do Milvus remoto (Collection).
        
        IMPORTANTE: Diferentes versões do Milvus podem retornar distâncias diferentes!
        """
        processed_results = []
        
        for hits in results:
            for hit in hits:
                try:
                    raw_distance = hit.distance
                    
                    # Detecta automaticamente o formato da distância
                    # Por algum motivo a distancia e a similaridade estão invertidas
                    # Me parece ter lido em algum lugar que no Milvus podia ser assim
                    # O que el chama de "distance" é a similaridade... Esta troca será feita
                    # dentro do método de nornalização.
                    cosine_distance, similarity_score = self._normalize_distance(raw_distance)
                    
                    logger.debug(f"REMOTE - Raw: {raw_distance:.6f}, "
                               f"Cosine: {cosine_distance:.6f}, "
                               f"Similarity: {similarity_score:.6f}")
                    
                    # Filtra por threshold
                    if similarity_score >= settings.similarity_threshold:
                        processed_results.append(UserSearchResult(
                            name=hit.entity.get("name"),
                            registration_number=hit.entity.get("registration_number"),
                            similarity_score=similarity_score,
                            distance=cosine_distance
                        ))
                        
                except Exception as e:
                    logger.error(f"Erro ao processar resultado remoto: {e}")
        
        return processed_results
    
    def _normalize_distance(self, raw_distance: float) -> tuple[float, float]:
        """
        Normaliza distância/score retornado pelo Milvus para formato consistente.
        
        O Milvus v2.5.x com métrica COSINE retorna distância cosseno no range [0, 2]:
        - 0.0: Vetores idênticos (máxima similaridade)
        - 1.0: Vetores ortogonais (similaridade neutra) 
        - 2.0: Vetores opostos (mínima similaridade)
        
        Esta função converte para:
        - cosine_distance: [0, 2] (mantém o formato original)
        - similarity_score: [0, 1] (onde 1.0 = idêntico, 0.0 = oposto)
        
        Args:
            raw_distance (float): Distância/score retornado pelo Milvus
            
        Returns:
            tuple[float, float]: (cosine_distance, similarity_score)
                - cosine_distance: Distância cosseno [0, 2]
                - similarity_score: Score de similaridade [0, 1]
                
        Raises:
            ValueError: Se raw_distance estiver fora do range esperado
            
        Examples:
            >>> # Vetores idênticos
            >>> cosine_dist, similarity = _normalize_distance(0.0)
            >>> print(f"Distance: {cosine_dist}, Similarity: {similarity}")
            Distance: 0.0, Similarity: 1.0
            
            >>> # Vetores com alguma diferença
            >>> cosine_dist, similarity = _normalize_distance(0.6406)
            >>> print(f"Distance: {cosine_dist}, Similarity: {similarity}")
            Distance: 0.6406, Similarity: 0.3594
        """
        
        # Valida range de entrada para distância cosseno
        if not (0.0 <= raw_distance <= 2.0):
            logger.warning(
                f"Distância fora do range esperado [0, 2]: {raw_distance}. "
                f"Milvus v{self.milvus_version} com COSINE deveria retornar [0, 2]"
            )
            
            # Tenta recuperar valores fora do range
            if raw_distance < 0.0:
                logger.warning("Distância negativa detectada, convertendo para 0.0")
                raw_distance = 0.0
            elif raw_distance > 2.0:
                logger.warning("Distância > 2.0 detectada, limitando a 2.0")
                raw_distance = 2.0
        
        # Conversão correta de distância cosseno para similaridade
        cosine_distance = raw_distance
        similarity_score = 1.0 - cosine_distance
        
        # Garante que similarity_score esteja no range [0, 1]
        similarity_score = max(0.0, min(1.0, similarity_score))
        
        logger.debug(
            f"Conversão - Raw: {raw_distance:.6f} -> "
            f"Cosine Distance: {cosine_distance:.6f}, "
            f"Similarity: {similarity_score:.6f}"
        )
        
        # Troca rezalizada entre distancia e similaridade, valores invertidos
        cosine_distance, similarity_score = similarity_score, cosine_distance
        return cosine_distance, similarity_score

    def get_debug_info(self) -> Dict:
        """
        Retorna informações de debug sobre a configuração atual.
        """
        try:
            info = {
                "milvus_version": self.milvus_version,
                "collection_name": self.collection_name,
                "dimension": self.dimension,
                "mode": "local" if settings.is_local_mode else "remote",
                "threshold": settings.similarity_threshold
            }
            
            if settings.is_local_mode:
                if self.client.has_collection(self.collection_name):
                    stats = self.client.get_collection_stats(self.collection_name)
                    info["total_records"] = stats.get("row_count", 0)
                    info["metric_type"] = "COSINE (local)"
            else:
                if utility.has_collection(self.collection_name):
                    info["total_records"] = self.collection.num_entities
                    # Tenta obter informações do índice
                    try:
                        index = self.collection.index(field_name="embedding")
                        info["index_type"] = index.index_type
                        info["metric_type"] = index.metric_type
                        info["index_params"] = index.params
                    except:
                        info["metric_type"] = "COSINE (presumido)"
            
            return info
            
        except Exception as e:
            logger.error(f"Erro ao obter debug info: {e}")
            return {"error": str(e)}
    
    def clear_all_data(self, confirm: bool = False) -> bool:
        """
        Elimina completamente a collection e recria do zero.
        PERIGOSO - Remove TODOS os dados!
        """
        if not confirm:
            raise ValueError(
                "Para eliminar e recriar a collection, você deve passar confirm=True. "
                "Esta operação é irreversível e remove TODOS os dados e índices!"
            )
        
        try:
            logger.warning("INICIANDO ELIMINAÇÃO COMPLETA DA COLLECTION...")
            
            if settings.is_local_mode:
                if self.client.has_collection(self.collection_name):
                    self.client.drop_collection(self.collection_name)
                    logger.info(f"Collection '{self.collection_name}' eliminada (modo local)")
            else:
                if utility.has_collection(self.collection_name):
                    try:
                        self.collection.release()
                        logger.info("Collection descarregada da memória")
                    except Exception as e:
                        logger.warning(f"Aviso ao descarregar: {e}")
                    
                    utility.drop_collection(self.collection_name)
                    logger.info(f"Collection '{self.collection_name}' eliminada (modo remoto)")
            
            # Recria a collection
            self._initialize_connection()    
            
            logger.warning("COLLECTION COMPLETAMENTE RECRIADA!")
            debug_info = self.get_debug_info()
            for key, value in debug_info.items():
                logger.info(f"{key}: {value}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao recriar collection: {e}")
            return False

# Instância global do serviço
milvus_service = MilvusService()