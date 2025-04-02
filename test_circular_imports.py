# Script simplificado para probar la importación de módulos y detectar importaciones circulares
import sys
import os

# Configurar el path para incluir el directorio raíz
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("== Prueba de resolución de importaciones circulares ==")

# Lista de módulos a probar
modules_to_test = [
    "common.config",
    "common.supabase",
    "common.cache",
    "common.context"
]

# Prueba 1: Importaciones básicas
print("\n1. IMPORTACIÓN DE MÓDULOS:")
for module_name in modules_to_test:
    try:
        print(f"Importando {module_name}...", end="")
        __import__(module_name)
        print(" OK")
    except Exception as e:
        print(f" ERROR: {str(e)}")

# Prueba 2: Importaciones específicas que causaban el problema
print("\n2. FUNCIONES ESPECÍFICAS QUE CAUSABAN PROBLEMAS:")

# Probar config.get_settings
try:
    print("Importando common.config.get_settings()...", end="")
    from common.config import get_settings
    print(" OK")
    print("Ejecutando get_settings()...", end="")
    settings = get_settings()
    print(" OK")
except Exception as e:
    print(f" ERROR: {str(e)}")

# Probar supabase.get_tenant_configurations
try:
    print("Importando common.supabase.get_tenant_configurations()...", end="")
    from common.supabase import get_tenant_configurations
    print(" OK")
    print("Ejecutando get_tenant_configurations()...", end="")
    configs = get_tenant_configurations()
    print(" OK")
except Exception as e:
    print(f" ERROR: {str(e)}")

# Prueba 3: Comprobar que ambos pueden importarse en el mismo script
print("\n3. PRUEBA DE INTERACCIÓN ENTRE MÓDULOS:")
try:
    print("Importando ambos módulos en el mismo script...")
    from common.config import get_settings
    from common.supabase import get_tenant_configurations, get_effective_configurations
    print("-> config.get_settings y supabase.get_tenant_configurations importados correctamente")
    
    print("Intentando ejecutar ambas funciones...")
    settings = get_settings()
    configs = get_tenant_configurations()
    print("-> Ambas funciones ejecutadas correctamente")
    
    print("\n¡ÉXITO! El problema de importación circular ha sido resuelto.")
    print("Los módulos pueden importarse y utilizarse sin conflictos.")
except Exception as e:
    print(f"ERROR: {str(e)}")
    print("El problema de importación circular NO ha sido resuelto completamente.")

print("\n== Fin de la prueba ==")
