import csv
import json
from datetime import datetime, timezone
from pathlib import Path

class CheckpointManager:
    """
    Gestor transaccional concurrente para asegurar la retención atómica
    de avances parciales por hilo de ejecución (worker). Previene la pérdida
    de datos ante caídas de servidor y evita el reprocesamiento redundante.
    """
    def __init__(self, worker_id: str = "worker_001", base_dir: Path | str | None = None):
        self.worker_id = str(worker_id).strip()
        if base_dir is None:
            self.base_dir = Path("reports/v37_manipulante2/rebuild_v1/checkpoints")
        else:
            self.base_dir = Path(base_dir)
            
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.base_dir / f"checkpoint_{self.worker_id}.csv"
        self.error_file = self.base_dir / f"errors_{self.worker_id}.csv"
        
        self._ensure_csv_headers()

    def _ensure_csv_headers(self):
        # Inicialización de cabeceras si el archivo físico no existe
        if not self.checkpoint_file.exists():
            with open(self.checkpoint_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["config_id", "partition", "phase", "status", "metrics_json", "timestamp", "input_hash"])
                
        if not self.error_file.exists():
            with open(self.error_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "config_id", "error_message"])

    def save_checkpoint(
        self,
        config_id: str,
        partition: str,
        phase: str,
        status: str,
        metrics: dict,
        input_hash: str
    ):
        """
        Consolida un avance de configuración en disco duro empleando escritura atómica
        mediante un archivo intermedio temporal y reemplazo del original para prevenir corrupciones.
        """
        status_clean = status.upper().strip()
        metrics_str = json.dumps(metrics, sort_keys=True)
        ts_iso = datetime.now(timezone.utc).isoformat()
        
        # Leemos el contenido existente para reescribirlo de forma atómica agregando la nueva fila
        rows = []
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    rows = list(reader)
            except Exception:
                # Si hubiera un sector mal cerrado, partimos de cabecera
                rows = [["config_id", "partition", "phase", "status", "metrics_json", "timestamp", "input_hash"]]
                
        if not rows or rows[0] != ["config_id", "partition", "phase", "status", "metrics_json", "timestamp", "input_hash"]:
            rows.insert(0, ["config_id", "partition", "phase", "status", "metrics_json", "timestamp", "input_hash"])
            
        rows.append([config_id, partition, phase, status_clean, metrics_str, ts_iso, input_hash])
        
        # Escritura atómica garantizada
        tmp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(tmp_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            
        # Reemplazo de archivo POSIX/Windows robusto
        tmp_file.replace(self.checkpoint_file)

    def log_error(self, config_id: str, error_message: str):
        """Asienta en un registro separado fallas o excepciones puras sin alterar completados."""
        ts_iso = datetime.now(timezone.utc).isoformat()
        with open(self.error_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([ts_iso, config_id, str(error_message)])

    def get_completed_configs(self) -> set[str]:
        """
        Escanea la totalidad de los archivos de checkpoints de los workers
        en el directorio base para consolidar las configuraciones ya procesadas exitosamente.
        Omite de forma estricta registros con estatus fallido o interrumpido.
        """
        completed = set()
        if not self.base_dir.exists():
            return completed
            
        for filepath in self.base_dir.glob("checkpoint_*.csv"):
            try:
                with open(filepath, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Si el archivo está corrupto o malformado, omitir la fila
                        if not row or "status" not in row or "config_id" not in row:
                            continue
                        if row["status"] == "COMPLETED":
                            completed.add(row["config_id"])
            except Exception:
                # Archivos ilegibles no bloquean la consolidación global
                continue
                
        return completed
