"""
Script para diagnosticar problemas de conectividad con Supabase desde Docker.
"""
import socket
import http.client
import json
import os
import sys

print("=== TEST DE CONECTIVIDAD A SUPABASE ===")

# 1. Prueba de resolución DNS
supabase_host = "zjauczcqzpggpaimgbqx.supabase.co"
try:
    print(f"\n1. Prueba de resolución DNS para: {supabase_host}")
    ip_address = socket.gethostbyname(supabase_host)
    print(f"✅ Resolución DNS exitosa: {supabase_host} -> {ip_address}")
except Exception as e:
    print(f"❌ Error en resolución DNS: {str(e)}")
    sys.exit(1)

# 2. Prueba de conexión a puerto 443
try:
    print(f"\n2. Prueba de conexión TCP a: {supabase_host}:443")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    result = s.connect_ex((supabase_host, 443))
    if result == 0:
        print(f"✅ Conexión TCP exitosa al puerto 443")
    else:
        print(f"❌ Conexión TCP fallida, código de error: {result}")
    s.close()
except Exception as e:
    print(f"❌ Error en prueba de conexión: {str(e)}")

# 3. Prueba de conexión HTTPS básica
try:
    print(f"\n3. Prueba de conexión HTTPS a: {supabase_host}")
    conn = http.client.HTTPSConnection(supabase_host, timeout=5)
    conn.request("GET", "/")
    response = conn.getresponse()
    print(f"✅ Conexión HTTPS exitosa: Status {response.status} {response.reason}")
    conn.close()
except Exception as e:
    print(f"❌ Error en conexión HTTPS: {str(e)}")

# 4. Mostrar variables de entorno relacionadas con Supabase
print("\n4. Variables de entorno de Supabase:")
print(f"SUPABASE_URL: {os.environ.get('SUPABASE_URL', 'No definida')}")
print(f"SUPABASE_KEY: {'Definida (oculta)' if os.environ.get('SUPABASE_KEY') else 'No definida'}")
print(f"SUPABASE_SERVICE_KEY: {'Definida (oculta)' if os.environ.get('SUPABASE_SERVICE_KEY') else 'No definida'}")

print("\n=== FIN DE PRUEBAS DE CONECTIVIDAD ===")
