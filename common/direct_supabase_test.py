"""
Test directo de conectividad a Supabase con resolución de DNS explícita
"""

import os
import socket
import requests
from dotenv import load_dotenv

load_dotenv()

# Obtener la URL de Supabase del archivo .env
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
if not SUPABASE_URL:
    print("Error: SUPABASE_URL no está configurada en el archivo .env")
    exit(1)

# Extraer el nombre de host de la URL
import re
hostname_match = re.search(r'https?://([^/:]+)', SUPABASE_URL)
if not hostname_match:
    print(f"Error: No se pudo extraer el nombre de host de {SUPABASE_URL}")
    exit(1)

hostname = hostname_match.group(1)

print(f"=== TEST DE CONECTIVIDAD BÁSICA A SUPABASE ===")
print(f"URL: {SUPABASE_URL}")
print(f"Hostname: {hostname}")

# 1. Prueba de resolución de DNS
print("\n--- Prueba de resolución DNS ---")
try:
    # Intentar forzar resolución IPv4
    print("Intentando resolución IPv4...")
    ipv4_info = socket.getaddrinfo(hostname, 443, socket.AF_INET, socket.SOCK_STREAM)
    if ipv4_info:
        for info in ipv4_info:
            addr_family, sock_type, proto, canon_name, sock_addr = info
            host, port = sock_addr
            print(f"✅ IPv4 resuelto: {host} (puerto {port})")
    
        # 2. Prueba de conectividad HTTP
        print("\n--- Prueba de conectividad HTTP ---")
        try:
            # Usar la IP directamente para evitar problemas de DNS
            ip = ipv4_info[0][4][0]
            # Construir URL con IP (manteniendo el protocolo)
            protocol = "https://" if SUPABASE_URL.startswith("https") else "http://"
            ip_url = f"{protocol}{ip}"
            
            print(f"Conectando a {ip_url} (con encabezado Host: {hostname})...")
            response = requests.get(
                ip_url,
                headers={
                    "Host": hostname,  # Importante: encabezado Host para TLS/SNI
                    "Content-Type": "application/json"
                },
                timeout=5,
                verify=False  # Desactivar verificación de certificado al usar IP directa
            )
            print(f"✅ Conexión HTTP exitosa: Código {response.status_code}")
        except Exception as e:
            print(f"❌ Error de conexión HTTP: {str(e)}")
except socket.gaierror as e:
    print(f"❌ Error de resolución DNS: {str(e)}")
    
    # Si falla IPv4, intentar con la familia predeterminada
    print("\nIntentando resolución de DNS con configuración predeterminada...")
    try:
        default_info = socket.getaddrinfo(hostname, 443)
        if default_info:
            for info in default_info:
                addr_family, sock_type, proto, canon_name, sock_addr = info
                if addr_family == socket.AF_INET:  # IPv4
                    host, port = sock_addr
                    print(f"✅ IPv4 resuelto (predeterminado): {host}")
                elif addr_family == socket.AF_INET6:  # IPv6
                    host, port, flow_info, scope_id = sock_addr
                    print(f"✅ IPv6 resuelto: {host}")
    except socket.gaierror as e2:
        print(f"❌ Error de resolución DNS (predeterminada): {str(e2)}")
except Exception as e:
    print(f"❌ Error inesperado: {str(e)}")

# 3. Sugerencia si hay problemas
if socket.has_ipv6:
    print("\n--- Información adicional ---")
    print("Este sistema tiene soporte para IPv6. Si hay problemas de conectividad,")
    print("podría ser útil forzar el uso de IPv4 para las conexiones a Supabase.")
    
# Proporcionar sugerencias prácticas para resolver el problema
print("\n--- Sugerencias de solución ---")
print("1. Verificar que los servidores DNS (8.8.8.8 y 1.1.1.1) están accesibles")
print("2. Añadir una entrada al archivo hosts para resolver explícitamente:")
print(f"   ejemplo: 123.123.123.123  {hostname}")
print("3. Asegurarse de que los firewalls permiten las conexiones salientes")
print("4. Para contenedores Docker, verificar la configuración de red")
