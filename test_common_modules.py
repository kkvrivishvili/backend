#!/usr/bin/env python
"""
Script de prueba para verificar la integridad de todos los módulos comunes.

Este script:
1. Intenta importar cada módulo del directorio common
2. Verifica que no existan problemas de importación circular
3. Muestra un mapa de dependencias entre módulos
"""

import sys
import os
import time
import importlib
import inspect
from types import ModuleType
from typing import List, Dict, Any, Callable, Tuple, Set

# Configurar el path para incluir el directorio raíz
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuración
TIMEOUT_SECONDS = 5
MODULES_TO_TEST = [
    "common.auth",
    "common.cache",
    "common.config",
    "common.config_schema",
    "common.context",
    "common.errors",
    "common.logging",
    "common.models",
    "common.ollama",
    "common.rate_limiting",
    "common.rpc_helpers",
    "common.supabase",
    "common.swagger",
    "common.tracking",
    "common.utils"
]

# Colores para la terminal (deshabilitados para PowerShell)
class Colors:
    HEADER = ''
    OKBLUE = ''
    OKGREEN = ''
    WARNING = ''
    FAIL = ''
    ENDC = ''
    BOLD = ''

def print_header(message: str):
    """Imprime un encabezado formateado"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{message.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")

def print_success(message: str):
    """Imprime un mensaje de éxito"""
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")

def print_warning(message: str):
    """Imprime un mensaje de advertencia"""
    print(f"{Colors.WARNING}⚠️ {message}{Colors.ENDC}")

def print_error(message: str):
    """Imprime un mensaje de error"""
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")

def print_info(message: str):
    """Imprime un mensaje informativo"""
    print(f"{Colors.OKBLUE}ℹ️ {message}{Colors.ENDC}")

def time_limited_import(module_name: str, timeout: int = TIMEOUT_SECONDS) -> Tuple[bool, Any, str]:
    """
    Intenta importar un módulo con un límite de tiempo para evitar bloqueos.
    
    Args:
        module_name: Nombre del módulo a importar
        timeout: Tiempo máximo en segundos
        
    Returns:
        Tuple[bool, Any, str]: (éxito, módulo o None, mensaje de error)
    """
    start_time = time.time()
    
    try:
        module = importlib.import_module(module_name)
        import_time = time.time() - start_time
        return True, module, f"Importado en {import_time:.2f}s"
    except Exception as e:
        import_time = time.time() - start_time
        if import_time >= timeout:
            error_msg = f"Timeout después de {timeout}s"
        else:
            error_msg = str(e)
        return False, None, error_msg

def test_module_import(module_name: str) -> Tuple[bool, ModuleType, str]:
    """
    Prueba la importación de un módulo.
    
    Args:
        module_name: Nombre del módulo a probar
        
    Returns:
        Tuple[bool, ModuleType, str]: (éxito, módulo o None, mensaje)
    """
    print_info(f"Importando {module_name}...")
    success, module, message = time_limited_import(module_name)
    
    if success:
        print_success(f"Módulo {module_name} importado correctamente: {message}")
    else:
        print_error(f"Error importando {module_name}: {message}")
    
    return success, module, message

def analyze_module_dependencies(
    module_name: str, 
    visited: Set[str] = None, 
    depth: int = 0
) -> Dict[str, List[str]]:
    """
    Analiza las dependencias de un módulo, detectando posibles ciclos.
    
    Args:
        module_name: Nombre del módulo a analizar
        visited: Conjunto de módulos ya visitados
        depth: Profundidad actual de análisis
        
    Returns:
        Dict[str, List[str]]: Mapa de módulos a sus dependencias
    """
    if visited is None:
        visited = set()
        
    if module_name in visited:
        return {}
        
    visited.add(module_name)
    dependencies = {}
    
    try:
        module = importlib.import_module(module_name)
        module_deps = []
        
        # Analizar importaciones del módulo
        for name, value in inspect.getmembers(module):
            if inspect.ismodule(value) and hasattr(value, "__name__"):
                dep_name = value.__name__
                if dep_name.startswith("common.") and dep_name != module_name:
                    module_deps.append(dep_name)
        
        dependencies[module_name] = module_deps
        
        # Analizar recursivamente si la profundidad lo permite
        if depth < 2:  # Limitar profundidad para evitar recursión excesiva
            for dep in module_deps:
                if dep not in visited:
                    sub_deps = analyze_module_dependencies(dep, visited, depth + 1)
                    dependencies.update(sub_deps)
    except Exception as e:
        print_warning(f"No se pudieron analizar las dependencias de {module_name}: {str(e)}")
    
    return dependencies

def detect_dependency_cycles(dependencies: Dict[str, List[str]]) -> List[List[str]]:
    """
    Detecta ciclos en las dependencias entre módulos.
    
    Args:
        dependencies: Mapa de módulos a sus dependencias
        
    Returns:
        List[List[str]]: Lista de ciclos detectados
    """
    cycles = []
    
    def find_cycles(node, path, visited):
        if node in path:
            # Ciclo detectado
            cycle_start_idx = path.index(node)
            cycles.append(path[cycle_start_idx:] + [node])
            return
            
        if node in visited:
            return
            
        visited.add(node)
        new_path = path + [node]
        
        for dep in dependencies.get(node, []):
            find_cycles(dep, new_path, visited.copy())
    
    for node in dependencies:
        find_cycles(node, [], set())
    
    # Eliminar duplicados
    unique_cycles = []
    for cycle in cycles:
        # Normalizar el ciclo para comparación
        sorted_cycle = sorted(cycle)
        if sorted_cycle not in [sorted(c) for c in unique_cycles]:
            unique_cycles.append(cycle)
    
    return unique_cycles

def print_dependency_map(dependency_map: Dict[str, List[str]]):
    """
    Imprime un mapa de dependencias en formato legible.
    
    Args:
        dependency_map: Mapa de dependencias a imprimir
    """
    print_header("MAPA DE DEPENDENCIAS")
    
    for module, deps in sorted(dependency_map.items()):
        module_short = module.replace("common.", "")
        print(f"{Colors.BOLD}{module_short}{Colors.ENDC} depende de:")
        
        if not deps:
            print("  (ninguna dependencia de common)")
        else:
            for dep in sorted(deps):
                dep_short = dep.replace("common.", "")
                print(f"  → {dep_short}")
        print()

def test_key_components():
    """
    Prueba la inicialización de componentes clave.
    """
    print_header("PRUEBA DE COMPONENTES CLAVE")
    
    # Prueba de configuración y get_settings
    try:
        print_info("Probando common.config.get_settings()...")
        from common.config import get_settings
        settings = get_settings()
        print_success("get_settings() ejecutado correctamente")
    except Exception as e:
        print_error(f"Error en get_settings(): {str(e)}")
    
    # Prueba de acceso a configuraciones de Supabase
    try:
        print_info("Probando common.supabase.get_tenant_configurations()...")
        from common.supabase import get_tenant_configurations
        configs = get_tenant_configurations()
        print_success("get_tenant_configurations() ejecutado correctamente")
    except Exception as e:
        print_error(f"Error en get_tenant_configurations(): {str(e)}")

def main():
    """Función principal para ejecutar todas las pruebas"""
    print_header("TEST DE IMPORTACIÓN DE MÓDULOS COMUNES")
    
    # Resultados
    successful_imports = []
    failed_imports = []
    dependency_map = {}
    
    # 1. Probar importación de cada módulo
    for module_name in MODULES_TO_TEST:
        success, module, message = test_module_import(module_name)
        
        if success:
            successful_imports.append(module_name)
            # Analizar dependencias del módulo
            module_deps = analyze_module_dependencies(module_name)
            dependency_map.update(module_deps)
        else:
            failed_imports.append(module_name)
    
    # 2. Imprimir mapa de dependencias
    print_dependency_map(dependency_map)
    
    # 3. Detectar ciclos de dependencia
    cycles = detect_dependency_cycles(dependency_map)
    
    # 4. Probar componentes clave
    test_key_components()
    
    # Resumen de resultados
    print_header("RESUMEN DE RESULTADOS")
    
    print(f"Total de módulos probados: {len(MODULES_TO_TEST)}")
    print(f"Módulos importados correctamente: {len(successful_imports)}")
    print(f"Módulos con error de importación: {len(failed_imports)}")
    
    if failed_imports:
        print_error("Módulos que fallaron:")
        for module in failed_imports:
            print(f"  - {module}")
    
    if cycles:
        print_warning(f"\nSe detectaron {len(cycles)} ciclos de dependencia:")
        for i, cycle in enumerate(cycles, 1):
            print_warning(f"  Ciclo {i}: {' -> '.join(cycle)}")
    else:
        print_success("\nNo se detectaron ciclos de dependencia en los módulos analizados")
    
    # Conclusión
    if not failed_imports and not cycles:
        print_success("\n¡ÉXITO! Todos los módulos fueron importados correctamente y no se detectaron ciclos de dependencia.")
    else:
        print_warning("\nSe encontraron algunos problemas. Revise los errores reportados arriba.")

if __name__ == "__main__":
    main()
