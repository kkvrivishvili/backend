#!/usr/bin/env python
"""Script para probar la importación de los módulos modificados"""

import sys
import os

# Configurar el path para incluir el directorio raíz
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("Probando importaciones...")

try:
    # Importar los módulos que antes causaban la importación circular
    print("Importando common.config...")
    from common.config import get_settings
    print("✅ Importación de common.config exitosa")
    
    print("\nImportando common.supabase...")
    from common.supabase import get_tenant_configurations, get_effective_configurations
    print("✅ Importación de common.supabase exitosa")
    
    print("\nProbando get_settings()...")
    settings = get_settings()
    print("✅ get_settings() ejecutado correctamente")
    
    print("\nProbando get_tenant_configurations()...")
    # Usamos None para que use el tenant por defecto
    configs = get_tenant_configurations(tenant_id=None)
    print("✅ get_tenant_configurations() ejecutado correctamente")
    
    print("\n¡Todo funcionó correctamente! El problema de importación circular ha sido resuelto.")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
except Exception as e:
    print(f"❌ Error general: {e}")
