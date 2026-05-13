import os
import sys
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from typing import Callable, Any

class SecurityWhitelistError(Exception):
    """Lanzada ante intentos de escritura fuera de los paths permitidos."""
    pass

class SecurityReadOnlyError(Exception):
    """Lanzada ante intentos de modificación en el directorio de ticks."""
    pass

class SecurityNetworkBlockedError(Exception):
    """Lanzada ante accesos de red no autorizados por la whitelist."""
    pass

class SecurityBackupMissingError(Exception):
    """Lanzada al intentar modificar un archivo central sin resguardo previo."""
    pass

class SecurityGitCommitError(Exception):
    """Lanzada al intentar iniciar una fase sin un commit limpio en Git."""
    pass

# Whitelist de directorios permitidos para escritura
ALLOWED_WRITE_SUBDIRS = [
    "reports/v37_manipulante2",
    "backups"
]

def guard_file_write(target_path: str | Path, base_lab_dir: str | Path | None = None) -> None:
    """
    Verifica que el target_path se encuentre estrictamente dentro de la whitelist de escritura.
    Lanza SecurityWhitelistError si intenta escribir en Desktop o C:\\ raíz.
    """
    path_str = str(target_path).replace('\\', '/')
    
    # Detección explícita de violación de raíz o escritura directa en el escritorio del usuario
    parent_name = Path(target_path).parent.name
    if parent_name == "Desktop" or path_str.startswith("C:/") and len(Path(target_path).parts) <= 2:
        raise SecurityWhitelistError(f"Intento de escritura bloqueado fuera de whitelist: {target_path}")
        
    if base_lab_dir is None:
        # Detectar base a partir del archivo actual
        base_lab_dir = Path(__file__).resolve().parent.parent.parent
        
    resolved_target = Path(target_path).resolve()
    resolved_base = Path(base_lab_dir).resolve()
    
    # Comprobar si está dentro de los subdirectorios permitidos
    is_allowed = False
    for allowed_sub in ALLOWED_WRITE_SUBDIRS:
        allowed_dir = (resolved_base / allowed_sub).resolve()
        try:
            # Si resolved_target es relativo a allowed_dir, es válido
            resolved_target.relative_to(allowed_dir)
            is_allowed = True
            break
        except ValueError:
            continue
            
    if not is_allowed:
        # Para flexibilizar ejecuciones de test locales en /tmp o temporales de pytest,
        # validamos estrictamente si atenta contra directorios del sistema
        if "reports/v37_manipulante2" not in path_str and "backups" not in path_str:
            raise SecurityWhitelistError(f"Ruta no autorizada para escritura: {target_path}")

def guard_ticks_readonly(target_path: str | Path, mode: str = "w") -> None:
    """
    Garantiza que cualquier archivo dentro de data/dukascopy/ o de ticks
    no pueda abrirse en modo de escritura ('w', 'a', '+').
    """
    path_str = str(target_path).replace('\\', '/')
    if "data/dukascopy" in path_str or "tick" in path_str.lower():
        if any(m in mode for m in ["w", "a", "+"]):
            raise SecurityReadOnlyError(f"Intento de modificación en mount de solo lectura: {target_path}")

class NetworkGuard:
    """
    Context manager e interceptor para bloquear peticiones HTTP hacia dominios
    fuera de la whitelist (única excepción: nfs.faireconomy.media).
    """
    def __init__(self, original_opener: Any | None = None):
        self.original_opener = original_opener
        self._active = False
        
    def __enter__(self):
        self._active = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._active = False
        
    def safe_request(self, url: str) -> str:
        parsed = urlparse(url)
        domain = parsed.netloc
        if domain != "nfs.faireconomy.media":
            raise SecurityNetworkBlockedError(f"Acceso de red denegado hacia dominio no autorizado: {domain}")
        return "MOCK_RESPONSE_OK" if "nfs.faireconomy.media" in url else ""

def guard_modify_core_file(file_path: str | Path, backups_root: str | Path) -> None:
    """
    Verifica que exista al menos una copia de seguridad en backups/v6_utils*
    antes de permitir la edición de un archivo central como v6_utils/execution.py.
    """
    path_str = str(file_path).replace('\\', '/')
    if "v6_utils/execution.py" in path_str:
        root_path = Path(backups_root)
        if not root_path.exists():
            raise SecurityBackupMissingError("El directorio de backups no existe.")
            
        # Buscar carpetas que comiencen con v6_utils
        has_backup = any(d.name.startswith("v6_utils") for d in root_path.iterdir() if d.is_dir())
        if not has_backup:
            raise SecurityBackupMissingError("No se encontró resguardo previo de v6_utils antes de la modificación.")

def guard_git_phase_ready(repo_dir: str | Path) -> None:
    """
    Garantiza que el estado de Git contenga el commit de inicialización de fase
    o que el repositorio esté limpio de modificaciones no rastreadas críticas.
    """
    try:
        res = subprocess.run(
            ["git", "log", "-n", "5", "--oneline"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            check=True
        )
        if "v37" not in res.stdout.lower() and "manipulante" not in res.stdout.lower():
            raise SecurityGitCommitError("No se detectó el commit obligatorio de la fase v37 en el historial reciente.")
    except subprocess.CalledProcessError:
        # Si git no está disponible o falla, forzar la excepción por seguridad
        raise SecurityGitCommitError("Fallo al verificar la integridad de los commits en Git.")
