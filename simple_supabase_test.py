"""
Script simplificado para probar la conexión básica a Supabase sin dependencias complejas.
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client

# Cargar variables de entorno
load_dotenv()

# Obtener credenciales de Supabase desde variables de entorno
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")

print("=== TEST SIMPLE DE CONECTIVIDAD SUPABASE ===")
print(f"URL: {supabase_url}")
print(f"Clave anon disponible: {'Sí' if supabase_key else 'No'}")
print(f"Clave de servicio disponible: {'Sí' if supabase_service_key else 'No'}")

# Probar conexión con clave anónima
print("\n--- Conexión con clave anónima ---")
try:
    supabase_anon = create_client(supabase_url, supabase_key)
    # Prueba simple: obtener la hora actual de Supabase
    response = supabase_anon.rpc('get_utc_time').execute()
    if response.data:
        print(f"✅ Conexión exitosa. Hora UTC de Supabase: {response.data}")
    else:
        print("❌ Sin error, pero no se obtuvo respuesta de datos.")
except Exception as e:
    print(f"❌ Error de conexión: {str(e)}")

# Si hay clave de servicio, probar también
if supabase_service_key:
    print("\n--- Conexión con clave de servicio ---")
    try:
        supabase_service = create_client(supabase_url, supabase_service_key)
        # Prueba simple
        response = supabase_service.rpc('get_utc_time').execute()
        if response.data:
            print(f"✅ Conexión exitosa. Hora UTC de Supabase: {response.data}")
        else:
            print("❌ Sin error, pero no se obtuvo respuesta de datos.")
    except Exception as e:
        print(f"❌ Error de conexión: {str(e)}")
