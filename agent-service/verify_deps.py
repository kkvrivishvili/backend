import pkg_resources
import sys

def check_compatibility(version):
    # Lista de paquetes a verificar
    packages = [
        f"langchain=={version}",
        f"langchain-core=={version}", 
        f"langchain-openai=={version}",
        f"langchain-community=={version}"
    ]

    # Intentar resolver las dependencias
    try:
        pkg_resources.require(packages)
        print(f"✅ Todas las dependencias son compatibles en la versión {version}")
        return True
    except pkg_resources.VersionConflict as e:
        print(f"❌ Conflicto de versiones para {version}: {e}")
        return False
    except pkg_resources.DistributionNotFound as e:
        print(f"❌ Paquete no encontrado para {version}: {e}")
        return False

# Comprobar compatibilidad para versiones específicas que podrían funcionar
versions_to_check = ["0.3.11", "0.3.10", "0.3.5", "0.3.0"]

for version in versions_to_check:
    print(f"\nVerificando compatibilidad para versión {version}:")
    if check_compatibility(version):
        print(f"✓ La versión {version} funciona para todos los paquetes")
        break
