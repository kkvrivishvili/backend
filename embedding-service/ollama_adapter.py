"""
Adaptador para integrar soporte de Ollama en el servicio de embeddings
"""
import os
from typing import Optional, List, Dict, Any
import logging
from ollama_wrapper import OllamaEmbeddings

logger = logging.getLogger(__name__)

def get_embedding_model(model_name: Optional[str] = None):
    """
    Retorna el modelo de embeddings apropiado según la configuración.
    Si USE_OLLAMA=true, utiliza el wrapper de Ollama.
    De lo contrario, utiliza OpenAI.
    """
    use_ollama = os.environ.get("USE_OLLAMA", "").lower() == "true"
    
    if use_ollama:
        logger.info(f"Usando embeddings de Ollama con modelo {model_name or 'nomic-embed-text'}")
        return OllamaEmbeddings(
            model_name=model_name or os.environ.get("EMBEDDING_MODEL", "nomic-embed-text"),
            base_url=os.environ.get("OLLAMA_API_URL", "http://localhost:11434")
        )
    else:
        # Importar OpenAI aquí para evitar errores si no está instalado
        from llama_index.embeddings.openai import OpenAIEmbedding
        logger.info(f"Usando embeddings de OpenAI con modelo {model_name or 'text-embedding-3-small'}")
        return OpenAIEmbedding(
            model=model_name or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        )
