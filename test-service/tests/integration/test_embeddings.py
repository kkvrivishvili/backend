"""
Tests para verificar el correcto funcionamiento del servicio de embeddings y el modelo nomic-embed-text.

Este módulo prueba:
1. La generación de embeddings con nomic-embed-text
2. La respuesta correcta desde el endpoint de embeddings
3. La dimensionalidad esperada de los vectores generados
"""

import pytest
import numpy as np
from typing import Dict, Any, List

@pytest.mark.asyncio
async def test_embeddings_endpoint(http_client, tenant_headers):
    """Verifica que el endpoint de embeddings genere vectores con las dimensiones correctas."""
    url = "http://embedding-service:8001/embeddings"
    
    # Datos de prueba
    test_data = {
        "texts": ["Este es un texto de prueba para verificar embeddings", 
                 "Otro texto diferente para asegurar que funciona correctamente"]
    }
    
    # Realizar solicitud
    response = await http_client.post(url, json=test_data, headers=tenant_headers)
    
    # Verificar respuesta exitosa
    assert response.status_code == 200
    data = response.json()
    
    # Verificar estructura de respuesta
    assert "embeddings" in data
    assert len(data["embeddings"]) == 2  # Dos textos enviados
    
    # Verificar dimensionalidad de los embeddings (nomic-embed-text genera vectores de 768 dimensiones)
    first_embedding = data["embeddings"][0]
    assert len(first_embedding) == 768
    
    # Verificar que son números reales válidos
    assert all(isinstance(val, float) for val in first_embedding)
    
    # Verificar normalización (opcional, depende de la implementación)
    vector = np.array(first_embedding)
    norm = np.linalg.norm(vector)
    assert 0.99 <= norm <= 1.01  # Aproximadamente normalizado

@pytest.mark.asyncio
async def test_embedding_model_info(http_client, tenant_headers):
    """Verifica que el servicio informe correctamente el modelo usado para embeddings."""
    url = "http://embedding-service:8001/status"
    
    # Realizar solicitud
    response = await http_client.get(url, headers=tenant_headers)
    
    # Verificar respuesta exitosa
    assert response.status_code == 200
    data = response.json()
    
    # Verificar que se reporta el modelo correcto
    assert "embedding_model" in data
    
    # Si estamos en modo Ollama, el modelo debe ser "nomic-embed-text"
    if "USE_OLLAMA" in data and data["USE_OLLAMA"] == True:
        assert data["embedding_model"] == "nomic-embed-text"
