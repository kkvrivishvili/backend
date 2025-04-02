"""
Tests para verificar la conexión correcta a Supabase y la funcionalidad del service role.

Este módulo prueba:
1. La correcta inicialización del cliente Supabase
2. El acceso a tablas con prefijo correcto usando get_table_name()
3. El funcionamiento del bypass RLS con la clave de servicio
"""

import os
import sys
import pytest
from typing import Dict, Any

# Imports desde common
from common.supabase import get_supabase_client, get_table_name

# Lista de tablas a verificar
TABLES_TO_CHECK = [
    "tenant_configurations", 
    "tenant_subscriptions", 
    "collections", 
    "conversations", 
    "document_chunks",
    "users"
]

def test_table_name_function():
    """Test para verificar que get_table_name() devuelve el nombre correcto con el prefijo adecuado."""
    # Tablas en esquema public
    assert get_table_name("users") == "public.users"
    assert get_table_name("tenants") == "public.tenants"
    assert get_table_name("public.users") == "public.users"
    
    # Tablas en esquema ai
    assert get_table_name("collections") == "ai.collections"
    assert get_table_name("conversations") == "ai.conversations"
    assert get_table_name("document_chunks") == "ai.document_chunks"
    assert get_table_name("tenant_subscriptions") == "ai.tenant_subscriptions"
    assert get_table_name("ai.collections") == "ai.collections"

def test_supabase_client_initialization():
    """Test para verificar que get_supabase_client() inicializa correctamente con ambas claves."""
    # Con clave anon
    client_anon = get_supabase_client(use_service_role=False)
    assert client_anon is not None
    
    # Con clave de servicio
    client_service = get_supabase_client(use_service_role=True)
    assert client_service is not None
    
    # Verificar que son instancias diferentes
    assert id(client_anon) != id(client_service)

@pytest.mark.asyncio
async def test_supabase_service_role_access():
    """
    Test para verificar que la clave de servicio permite el acceso a tablas 
    con políticas RLS que normalmente bloquearían al usuario anónimo.
    """
    # Inicializar clientes
    client_anon = get_supabase_client(use_service_role=False)
    client_service = get_supabase_client(use_service_role=True)
    
    # Probar lectura de tablas con ambos clientes
    for table in TABLES_TO_CHECK:
        table_name = get_table_name(table)
        
        # Con cliente service role debe funcionar
        try:
            result_service = client_service.table(table_name).select("*").limit(1).execute()
            # Si llegamos aquí, no hubo excepciones - el acceso fue exitoso
            assert True
        except Exception as e:
            # Si hay excepción con servicio role, reportamos el error
            pytest.fail(f"Error con service role al acceder a {table_name}: {str(e)}")

        # Con cliente anon puede fallar por RLS, pero no lo hacemos fallar
        try:
            result_anon = client_anon.table(table_name).select("*").limit(1).execute()
            # Acceso exitoso con anon (tabla pública o RLS permite access)
            pass
        except Exception:
            # Error esperado para tablas con RLS restrictivas
            pass

@pytest.mark.skip(reason="Solo ejecutar manualmente para verificar estructura de tablas")
def test_print_table_schemas():
    """Test para imprimir la estructura de las tablas en Supabase (solo para referencia)."""
    client = get_supabase_client(use_service_role=True)
    
    for table in TABLES_TO_CHECK:
        table_name = get_table_name(table)
        try:
            # Obtener un registro para ver los campos
            result = client.table(table_name).select("*").limit(1).execute()
            if result.data:
                print(f"\nEstructura de {table_name}:")
                for key in result.data[0].keys():
                    print(f"  - {key}")
            else:
                print(f"\n{table_name}: No hay datos para mostrar estructura")
        except Exception as e:
            print(f"\nError al verificar {table_name}: {str(e)}")
