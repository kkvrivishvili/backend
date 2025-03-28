"""
Adaptador para integrar soporte de Ollama en el servicio de consulta
"""
import os
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_llm_model(model_name: Optional[str] = None):
    """
    Retorna el modelo LLM apropiado según la configuración.
    Si USE_OLLAMA=true, utiliza el modelo de Ollama.
    De lo contrario, utiliza OpenAI.
    """
    use_ollama = os.environ.get("USE_OLLAMA", "").lower() == "true"
    
    if use_ollama:
        # Importar los modelos necesarios de llama_index para Ollama
        from llama_index.llms import Ollama
        
        model = model_name or os.environ.get("LLM_MODEL", "llama3.2:1b")
        base_url = os.environ.get("OLLAMA_API_URL", "http://ollama:11434")
        logger.info(f"Usando LLM de Ollama con modelo {model} en {base_url}")
        
        return Ollama(
            model=model,
            base_url=base_url,
            temperature=0.7,
        )
    else:
        # Usar OpenAI
        from llama_index.llms.openai import OpenAI
        
        model = model_name or os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        logger.info(f"Usando LLM de OpenAI con modelo {model}")
        
        return OpenAI(
            model=model,
            temperature=0.7,
        )
