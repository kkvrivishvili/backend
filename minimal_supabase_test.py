"""
Prueba mínima de conectividad a Supabase
"""
import os
import socket
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Obtener URL de Supabase
supabase_url = os.environ.get('SUPABASE_URL', '')
print(f"URL de Supabase: {supabase_url}")

# Extraer el dominio
import re
hostname_match = re.search(r'https?://([^/:]+)', supabase_url)
if hostname_match:
    hostname = hostname_match.group(1)
    print(f"Dominio: {hostname}")
    
    # Intentar resolver el dominio
    try:
        ip_address = socket.gethostbyname(hostname)
        print(f"Resolución exitosa: {hostname} -> {ip_address}")
    except socket.gaierror as e:
        print(f"Error al resolver DNS: {e}")
else:
    print("No se pudo extraer el dominio de la URL de Supabase")
