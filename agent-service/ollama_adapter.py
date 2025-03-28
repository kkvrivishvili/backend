"""
Adaptador para integrar soporte de Ollama en el servicio de agente
"""
import os
from typing import Optional, Dict, Any, List
import logging
from langchain_community.llms import Ollama
from langchain_core.language_models import BaseLLM

logger = logging.getLogger(__name__)

def get_llm_model(model_name: Optional[str] = None) -> BaseLLM:
    """
    Retorna el modelo LLM apropiado según la configuración.
    Si USE_OLLAMA=true, utiliza el modelo de Ollama.
    De lo contrario, utiliza OpenAI.
    """
    use_ollama = os.environ.get("USE_OLLAMA", "").lower() == "true"
    
    if use_ollama:
        # Usar modelo de Ollama
        model = model_name or os.environ.get("LLM_MODEL", "llama3.2:1b")
        base_url = os.environ.get("OLLAMA_API_URL", "http://ollama:11434")
        logger.info(f"Usando LLM de Ollama con modelo {model} en {base_url}")
        
        return Ollama(
            model=model,
            base_url=base_url,
            temperature=0.7,
        )
    else:
        # Importar OpenAI aquí para evitar errores si no está instalado
        from langchain_openai import ChatOpenAI
        model = model_name or os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        logger.info(f"Usando LLM de OpenAI con modelo {model}")
        
        return ChatOpenAI(
            model=model,
            temperature=0.7,
        )
