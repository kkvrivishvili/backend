"""
Wrapper para modelos de embeddings de Ollama
"""
import httpx
import numpy as np
from typing import List, Optional, Any, Dict
import os
import logging

logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    """
    Clase para generar embeddings usando modelos de Ollama
    """
    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Inicializa el cliente de embeddings de Ollama
        
        Args:
            model_name: Nombre del modelo de embeddings a usar
            base_url: URL base de la API de Ollama (por defecto: http://localhost:11434)
            dimensions: Dimensiones deseadas para los embeddings (si el modelo lo soporta)
        """
        self.model_name = model_name
        self.base_url = base_url or os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
        self.dimensions = dimensions
        logger.info(f"Inicializando OllamaEmbeddings con modelo {model_name} en {self.base_url}")
    
    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos
        
        Args:
            texts: Lista de textos para generar embeddings
            
        Returns:
            Lista de embeddings (vectores)
        """
        embeddings = []
        
        async with httpx.AsyncClient() as client:
            for text in texts:
                # La API de Ollama espera un JSON con el texto a embeber
                try:
                    response = await client.post(
                        f"{self.base_url}/api/embeddings",
                        json={"model": self.model_name, "prompt": text}
                    )
                    response.raise_for_status()
                    data = response.json()
                    # Extraer el embedding del resultado
                    embedding = data.get("embedding", [])
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error al obtener embedding de Ollama: {str(e)}")
                    # En caso de error, devolver un vector de ceros
                    if self.dimensions:
                        embeddings.append([0.0] * self.dimensions)
                    else:
                        # Si no sabemos las dimensiones, usar un tamaño estándar
                        embeddings.append([0.0] * 768)
        
        return embeddings
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para documentos
        
        Args:
            texts: Lista de textos para generar embeddings
            
        Returns:
            Lista de embeddings (vectores)
        """
        return await self._embed_texts(texts)
    
    async def embed_query(self, text: str) -> List[float]:
        """
        Genera embedding para una consulta
        
        Args:
            text: Texto de la consulta
            
        Returns:
            Vector de embedding
        """
        embeddings = await self._embed_texts([text])
        return embeddings[0] if embeddings else []
        
    # Métodos adicionales para compatibilidad con CachedOpenAIEmbedding
    async def get_embedding(self, text: str) -> List[float]:
        """
        Genera embedding para un texto
        
        Args:
            text: Texto para generar embedding
            
        Returns:
            Vector de embedding
        """
        return await self.embed_query(text)
        
    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos
        
        Args:
            texts: Lista de textos para generar embeddings
            
        Returns:
            Lista de embeddings (vectores)
        """
        return await self.embed_documents(texts)
