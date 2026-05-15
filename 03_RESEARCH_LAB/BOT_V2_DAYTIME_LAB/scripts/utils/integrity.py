import os
import json
import logging
from pathlib import Path
from datetime import datetime

class AtomicSingleWriter:
    """
    Garantiza que solo un proceso escriba en un recurso compartido.
    Utiliza bloqueos atómicos a nivel de OS (O_EXCL).
    """
    def __init__(self, lock_path: Path, run_id: str, pid: int):
        self.lock_path = Path(lock_path)
        self.run_id = run_id
        self.pid = pid
        self.is_locked = False

    def acquire(self, metadata: dict = None) -> bool:
        """Intenta adquirir el bloqueo de forma atómica."""
        if self.is_locked:
            return True
            
        try:
            # Crear directorio de locks si no existe
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            
            # os.O_EXCL garantiza que la creación falle si el archivo ya existe (Atómico)
            fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            f = os.fdopen(fd, 'w')
            try:
                info = {
                    "run_id": self.run_id,
                    "pid": self.pid,
                    "timestamp": datetime.now().isoformat(),
                    "metadata": metadata or {}
                }
                json.dump(info, f, indent=4)
            finally:
                f.close()
            
            self.is_locked = True
            return True
        except FileExistsError:
            return False
        except Exception as e:
            logging.error(f"Error adquiriendo lock: {e}")
            return False

    def release(self):
        """Libera el bloqueo de forma segura."""
        if not self.is_locked:
            return
            
        try:
            if self.lock_path.exists():
                # Verificar que el lock sea nuestro antes de borrar
                should_remove = False
                with open(self.lock_path, 'r') as f:
                    data = json.load(f)
                    if data.get("run_id") == self.run_id:
                        should_remove = True
                    else:
                        logging.warning(f"No se puede liberar un lock que pertenece a otro RunID: {data.get('run_id')}")
                
                if should_remove:
                    os.remove(self.lock_path)
            self.is_locked = False
        except Exception as e:
            logging.error(f"Error liberando lock: {e}")

    def validate_stale(self, timeout_seconds=3600) -> bool:
        """
        Opcional: Verifica si un lock es antiguo (stale).
        Implementación básica por timestamp.
        """
        if not self.lock_path.exists():
            return True
        try:
            with open(self.lock_path, 'r') as f:
                data = json.load(f)
                ts = datetime.fromisoformat(data["timestamp"])
                if (datetime.now() - ts).total_seconds() > timeout_seconds:
                    return True
            return False
        except:
            return True
