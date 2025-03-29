"""
Módulo centralizado para todas las integraciones con Ollama.
Incluye clases para embeddings y LLM, así como funciones auxiliares.
"""

import httpx
import json
import logging
import os
from typing import List, Optional, Dict, Any, Union, Callable, AsyncGenerator

# Importación de configuración
from common.config import get_settings

# Configuración
settings = get_settings()
logger = logging.getLogger(__name__)

class OllamaEmbeddings:
    """
    Clase para generar embeddings usando modelos de Ollama.
    Implementa una interfaz compatible con las APIs de embedding comunes.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        """
        Inicializa el cliente de embeddings de Ollama
        
        Args:
            model_name: Nombre del modelo de embeddings a usar
            base_url: URL base de la API de Ollama
            dimensions: Dimensiones deseadas para los embeddings (si el modelo lo soporta)
        """
        self.model_name = model_name or settings.default_ollama_embedding_model
        self.base_url = base_url or settings.ollama_api_url
        self.dimensions = dimensions or settings.default_embedding_dimension
        
        logger.info(f"Inicializando OllamaEmbeddings con modelo {self.model_name} en {self.base_url}")
    
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
                        json={"model": self.model_name, "prompt": text},
                        timeout=30.0  # Timeout aumentado para modelos grandes
                    )
                    response.raise_for_status()
                    data = response.json()
                    # Extraer el embedding del resultado
                    embedding = data.get("embedding", [])
                    embeddings.append(embedding)
                except Exception as e:
                    logger.error(f"Error al obtener embedding de Ollama: {str(e)}")
                    # En caso de error, devolver un vector de ceros
                    embeddings.append([0.0] * self.dimensions)
        
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
        
    # Métodos adicionales para compatibilidad con diferentes APIs
    async def get_embedding(self, text: str) -> List[float]:
        """
        Genera embedding para un texto (compatible con LlamaIndex)
        
        Args:
            text: Texto para generar embedding
            
        Returns:
            Vector de embedding
        """
        return await self.embed_query(text)
        
    async def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para una lista de textos (compatible con LlamaIndex)
        
        Args:
            texts: Lista de textos para generar embeddings
            
        Returns:
            Lista de embeddings (vectores)
        """
        return await self.embed_documents(texts)


class OllamaLLM:
    """
    Clase para generar respuestas usando modelos LLM de Ollama.
    Implementa una interfaz similar a los LLMs comunes.
    """
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Inicializa el cliente LLM de Ollama
        
        Args:
            model_name: Nombre del modelo LLM a usar
            base_url: URL base de la API de Ollama
            temperature: Temperatura para la generación (0.0-1.0)
            max_tokens: Número máximo de tokens a generar
            stop_sequences: Secuencias donde detener la generación
            **kwargs: Parámetros adicionales para la API de Ollama
        """
        self.model_name = model_name or settings.default_ollama_llm_model
        self.base_url = base_url or settings.ollama_api_url
        self.temperature = temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.stop_sequences = stop_sequences or []
        self.additional_params = kwargs
        
        logger.info(f"Inicializando OllamaLLM con modelo {self.model_name} en {self.base_url}")
    
    async def _generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Genera texto usando el modelo LLM de Ollama
        
        Args:
            prompt: Texto de entrada para generar la respuesta
            system_prompt: Instrucción del sistema (opcional)
            stream: Si es True, transmite la respuesta token por token
            
        Returns:
            Texto generado o un generador de tokens si stream=True
        """
        # Construcción de la solicitud para la API de Ollama
        request_json = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        # Añadir system prompt si está presente
        if system_prompt:
            request_json["system"] = system_prompt
        
        # Añadir stop sequences si están presentes
        if self.stop_sequences:
            request_json["options"]["stop"] = self.stop_sequences
            
        # Añadir parámetros adicionales
        for key, value in self.additional_params.items():
            if key not in request_json["options"]:
                request_json["options"][key] = value
                
        if stream:
            # Streaming de respuesta
            return self._stream_response(request_json)
        else:
            # Respuesta completa
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json=request_json,
                        timeout=60.0  # Timeout generoso para generaciones largas
                    )
                    response.raise_for_status()
                    data = response.json()
                    return data.get("response", "")
            except Exception as e:
                logger.error(f"Error al obtener respuesta de Ollama: {str(e)}")
                return f"Error en la generación: {str(e)}"
    
    async def _stream_response(self, request_json: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Transmite la respuesta del modelo token por token
        
        Args:
            request_json: JSON de solicitud para la API de Ollama
            
        Yields:
            Tokens generados uno por uno
        """
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", f"{self.base_url}/api/generate", 
                                       json=request_json, timeout=60.0) as response:
                    response.raise_for_status()
                    
                    # Procesar la respuesta línea por línea
                    buffer = ""
                    async for chunk in response.aiter_text():
                        buffer += chunk
                        
                        # Buscar líneas completas JSON
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            if line.strip():
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        yield data["response"]
                                except json.JSONDecodeError:
                                    logger.warning(f"Error al decodificar JSON: {line}")
            except Exception as e:
                logger.error(f"Error al transmitir respuesta de Ollama: {str(e)}")
                yield f"Error en la generación: {str(e)}"
    
    async def predict(self, prompt: str, **kwargs) -> str:
        """
        Genera una predicción basada en un prompt (compatible con LangChain)
        
        Args:
            prompt: Texto de entrada
            **kwargs: Parámetros adicionales
            
        Returns:
            Texto generado
        """
        system_prompt = kwargs.get("system_prompt")
        return await self._generate(prompt, system_prompt=system_prompt, stream=False)
    
    async def apredict(self, prompt: str, **kwargs) -> str:
        """
        Versión asíncrona de predict (compatible con LangChain)
        """
        return await self.predict(prompt, **kwargs)
    
    async def agenerate(
        self, 
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Genera respuestas basadas en mensajes (compatible con LangChain)
        
        Args:
            messages: Lista de mensajes en formato LangChain
            **kwargs: Parámetros adicionales
            
        Returns:
            Respuesta generada
        """
        # Convertir mensajes al formato prompt + system
        system_content = None
        user_content = []
        
        for msg in messages:
            if msg.get("type") == "system" or msg.get("role") == "system":
                system_content = msg.get("content", "")
            elif msg.get("type") == "human" or msg.get("role") == "user":
                user_content.append(msg.get("content", ""))
        
        # Si no hay contenido del sistema en los mensajes, usar el de kwargs
        if not system_content and "system_prompt" in kwargs:
            system_content = kwargs["system_prompt"]
            
        # Combinar todo el contenido del usuario
        prompt = "\n".join(user_content)
        
        # Generar respuesta
        response_text = await self._generate(
            prompt=prompt, 
            system_prompt=system_content,
            stream=False
        )
        
        # Devolver en formato similar a OpenAI
        return {
            "generations": [
                {
                    "text": response_text,
                    "message": {"role": "assistant", "content": response_text}
                }
            ]
        }


def get_embedding_model(model_name: Optional[str] = None, **kwargs):
    """
    Retorna el modelo de embeddings apropiado según la configuración.
    
    Args:
        model_name: Nombre opcional del modelo a utilizar
        **kwargs: Parámetros adicionales para el modelo
        
    Returns:
        Un modelo de embeddings compatible
    """
    if settings.use_ollama:
        # Usar el modelo de Ollama de este módulo
        return OllamaEmbeddings(
            model_name=model_name or settings.default_ollama_embedding_model,
            base_url=kwargs.get("base_url", settings.ollama_api_url),
            dimensions=kwargs.get("dimensions", settings.default_embedding_dimension)
        )
    else:
        # Usar OpenAI
        try:
            # Intentar importar desde llama_index primero
            from llama_index.embeddings.openai import OpenAIEmbedding
            
            logger.info(f"Usando embeddings de OpenAI con modelo {model_name or settings.default_openai_embedding_model}")
            
            return OpenAIEmbedding(
                model=model_name or settings.default_openai_embedding_model,
                api_key=settings.openai_api_key,
                embed_batch_size=settings.embedding_batch_size
            )
        except ImportError:
            # Si no está disponible, usar langchain
            logger.warning("No se pudo importar desde llama_index, usando langchain")
            from langchain_openai import OpenAIEmbeddings
            
            return OpenAIEmbeddings(
                model=model_name or settings.default_openai_embedding_model,
                openai_api_key=settings.openai_api_key
            )


def get_llm_model(model_name: Optional[str] = None, **kwargs):
    """
    Retorna el modelo LLM apropiado según la configuración.
    
    Args:
        model_name: Nombre opcional del modelo a utilizar
        **kwargs: Parámetros adicionales para el modelo
        
    Returns:
        Un modelo LLM compatible
    """
    if settings.use_ollama:
        # Usar el modelo de Ollama de este módulo
        return OllamaLLM(
            model_name=model_name or settings.default_ollama_llm_model,
            base_url=kwargs.get("base_url", settings.ollama_api_url),
            temperature=kwargs.get("temperature", settings.llm_temperature),
            max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens)
        )
    else:
        # Usar OpenAI
        try:
            # Intentar importar desde langchain primero
            from langchain_openai import ChatOpenAI
            
            logger.info(f"Usando LLM de OpenAI con modelo {model_name or settings.default_openai_llm_model}")
            
            return ChatOpenAI(
                model=model_name or settings.default_openai_llm_model,
                openai_api_key=settings.openai_api_key,
                temperature=kwargs.get("temperature", settings.llm_temperature),
                max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens)
            )
        except ImportError:
            # Si no está disponible, intentar con llama_index
            try:
                from llama_index.llms.openai import OpenAI
                
                return OpenAI(
                    model=model_name or settings.default_openai_llm_model,
                    api_key=settings.openai_api_key,
                    temperature=kwargs.get("temperature", settings.llm_temperature),
                    max_tokens=kwargs.get("max_tokens", settings.llm_max_tokens)
                )
            except ImportError:
                logger.error("No se pudo importar modelo OpenAI desde ninguna librería")
                raise ImportError("No se pudo importar ningún modelo OpenAI compatible")


def is_using_ollama() -> bool:
    """
    Retorna si se está usando Ollama según la configuración.
    
    Returns:
        bool: True si se está usando Ollama, False si se está usando OpenAI
    """
    return settings.use_ollama
