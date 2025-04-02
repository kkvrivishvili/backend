#!/usr/bin/env python
"""
Script centralizado para ejecutar todas las pruebas del sistema.

Este script puede ejecutarse directamente o ser llamado desde el servicio de test.
Soporta diferentes modos de ejecución:
- Pruebas de infraestructura (database, redis, etc.)
- Pruebas de integración entre servicios
- Pruebas unitarias de componentes específicos
- Todas las pruebas juntas

Uso:
    python run_tests.py [--category <category>] [--service <service>] [--verbose]
    
    Categorías disponibles:
    - infrastructure: Pruebas de infraestructura (Supabase, Redis, etc.)
    - integration: Pruebas de integración entre servicios
    - unit: Pruebas unitarias de componentes aislados
    - all: Todas las pruebas (por defecto)
    
    Servicios específicos:
    - embedding-service: Pruebas del servicio de embeddings
    - query-service: Pruebas del servicio de consultas
    - agent-service: Pruebas del servicio de agentes
    - all: Todos los servicios (por defecto)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Asegurarnos que estamos en el directorio correcto para importaciones
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

def parse_args():
    """Procesa los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Ejecuta pruebas organizadas del sistema")
    parser.add_argument(
        "--category", 
        choices=["infrastructure", "integration", "unit", "all"], 
        default="all",
        help="Categoría de pruebas a ejecutar"
    )
    parser.add_argument(
        "--service", 
        choices=["embedding-service", "query-service", "agent-service", "all"], 
        default="all",
        help="Servicio específico a probar"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Mostrar salida detallada"
    )
    parser.add_argument(
        "--output", "-o",
        help="Archivo de salida para resultados (formato JSON)"
    )
    return parser.parse_args()

def run_pytest(test_path, verbose=False):
    """Ejecuta pytest en el path especificado y devuelve los resultados."""
    print(f"\n=== Ejecutando pruebas en: {test_path} ===")
    
    # Construir comando de pytest
    cmd = ["pytest", test_path, "-v"] if verbose else ["pytest", test_path]
    
    # Añadir opción para generar reporte JSON si se solicita salida
    if args.output:
        cmd.extend(["--json-report", "--json-report-file", args.output])
    
    # Ejecutar pytest
    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - start_time
    
    # Mostrar resultados
    if verbose or result.returncode != 0:
        print(result.stdout)
        if result.stderr:
            print("Errores:", result.stderr)
    
    # Extraer estadísticas básicas
    success = result.returncode == 0
    
    # Devolver información de la ejecución
    return {
        "success": success,
        "path": test_path,
        "duration": duration,
        "returncode": result.returncode,
        "output": result.stdout if verbose else ""
    }

def generate_report(results):
    """Genera un informe consolidado de todas las pruebas."""
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    total_duration = sum(r["duration"] for r in results)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "successful_tests": successful_tests,
            "failed_tests": total_tests - successful_tests,
            "success_rate": f"{(successful_tests / total_tests) * 100:.1f}%" if total_tests > 0 else "N/A",
            "total_duration": f"{total_duration:.2f} segundos"
        },
        "results": results
    }
    
    # Guardar informe en archivo si se especificó
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
    
    # Mostrar resumen
    print("\n=== RESUMEN DE PRUEBAS ===")
    print(f"Total de pruebas ejecutadas: {total_tests}")
    print(f"Pruebas exitosas: {successful_tests}")
    print(f"Pruebas fallidas: {total_tests - successful_tests}")
    print(f"Tasa de éxito: {(successful_tests / total_tests) * 100:.1f}%" if total_tests > 0 else "N/A")
    print(f"Duración total: {total_duration:.2f} segundos")
    print(f"Reporte completo guardado en: {args.output}" if args.output else "")
    
    return report

if __name__ == "__main__":
    # Obtener argumentos
    args = parse_args()
    verbose = args.verbose
    
    # Determinar paths de test según categoría y servicio
    test_base_path = os.path.join(script_dir, "tests")
    all_results = []
    
    if args.category == "all":
        # Ejecutar todas las categorías
        categories = ["infrastructure", "integration", "unit"]
    else:
        # Ejecutar solo la categoría especificada
        categories = [args.category]
    
    # Ejecutar pruebas para cada categoría
    for category in categories:
        category_path = os.path.join(test_base_path, category)
        
        # Verificar si el directorio existe
        if not os.path.exists(category_path):
            print(f"Advertencia: Directorio {category_path} no existe, saltando...")
            continue
        
        # Ejecutar pruebas según el servicio especificado
        if args.service == "all":
            # Ejecutar todas las pruebas de la categoría
            result = run_pytest(category_path, verbose)
            all_results.append(result)
        else:
            # Intentar encontrar pruebas específicas para el servicio
            service_specific_path = os.path.join(category_path, f"test_{args.service.replace('-', '_')}.py")
            if os.path.exists(service_specific_path):
                result = run_pytest(service_specific_path, verbose)
                all_results.append(result)
            else:
                # Buscar cualquier archivo que pueda contener pruebas para el servicio
                import glob
                service_files = glob.glob(os.path.join(category_path, f"*{args.service.replace('-', '_')}*.py"))
                if service_files:
                    for service_file in service_files:
                        result = run_pytest(service_file, verbose)
                        all_results.append(result)
                else:
                    print(f"No se encontraron pruebas específicas para {args.service} en {category}")
    
    # Generar reporte consolidado
    report = generate_report(all_results)
    
    # Establecer código de salida según resultados
    if report["summary"]["failed_tests"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)
