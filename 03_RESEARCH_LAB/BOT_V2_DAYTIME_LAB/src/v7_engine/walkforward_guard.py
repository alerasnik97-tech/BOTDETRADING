import hashlib
import json
from pathlib import Path
from dataclasses import dataclass, asdict

class MethodologicalContaminationError(Exception):
    """Excepción crítica levantada incondicionalmente ante cualquier violación del flujo walk-forward."""
    pass

@dataclass
class FrozenCandidate:
    candidate_id: str
    parameters: dict[str, any]
    validation_pf: float
    hash_signature: str = ""

    def compute_hash(self) -> str:
        # Serialización canónica predecible para garantizar consistencia del hash
        param_str = json.dumps(self.parameters, sort_keys=True)
        return hashlib.sha256(f"{self.candidate_id}:{param_str}:{self.validation_pf:.4f}".encode()).hexdigest()

class WalkForwardGuard:
    """
    Guarda inmutable de flujo de trabajo (Tarea F) para garantizar aislamiento
    metodológico estricto y prevenir contaminación de la partición de prueba (TEST).
    Auditado en paralelo: rediseñado para depender unívocamente de la verificación física
    en disco duro (FROZEN_CANDIDATE.json y FROZEN_CANDIDATE_SHA256.txt) erradicando la
    vulnerabilidad de autorizaciones basadas exclusivamente en memoria volátil.
    """
    def __init__(self, current_phase: str = "TRAIN", storage_dir: Path | str | None = None):
        self.current_phase = current_phase.upper().strip()
        if self.current_phase not in ["TRAIN", "VALIDATION", "TEST"]:
            raise ValueError(f"Fase metodológica desconocida: {self.current_phase}")
            
        if storage_dir is None:
            self.storage_dir = Path("reports/v37_manipulante2/rebuild_v1/validation")
        else:
            self.storage_dir = Path(storage_dir)
            
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.storage_dir / "FROZEN_CANDIDATE.json"
        self.sha_path = self.storage_dir / "FROZEN_CANDIDATE_SHA256.txt"
        
        # Referencia local en memoria retenida solo para compatibilidad de firmas,
        # pero inhabilitada como fuente de autorización en TEST.
        self.frozen_candidate: FrozenCandidate | None = None

    def set_phase(self, phase: str):
        self.current_phase = phase.upper().strip()

    def authorize_ranking(self, num_candidates: int) -> bool:
        """Autoriza el ordenamiento y selección de top candidates solo en fases tempranas."""
        if self.current_phase == "TEST":
            raise MethodologicalContaminationError(
                "Operación de ranking y selección múltiple estrictamente prohibida en la fase TEST."
            )
        return True

    def authorize_search_space_iteration(self) -> bool:
        """Impide la reexploración o reingreso al search space durante la prueba final."""
        if self.current_phase == "TEST":
            raise MethodologicalContaminationError(
                "Iteración o consulta del espacio de búsqueda prohibida en la fase TEST."
            )
        return True

    def _compute_file_sha256(self, filepath: Path) -> str:
        """Calcula el hash SHA256 canónico desde el contenido físico del archivo en disco."""
        return hashlib.sha256(filepath.read_bytes()).hexdigest()

    def select_frozen_candidate(self, candidate_id: str, parameters: dict[str, any], val_pf: float) -> FrozenCandidate:
        """
        Promueve una única candidata ganadora desde VALIDATION hacia TEST,
        escribiendo incondicionalmente a disco su JSON y su firma SHA256 externa.
        Bloquea cualquier intento de sobrescritura o promoción secundaria.
        """
        if self.current_phase != "VALIDATION":
            raise MethodologicalContaminationError(
                f"Selección de candidata congelada solo permitida en VALIDATION. Fase actual: {self.current_phase}"
            )
            
        # Regla 1 y 2: VALIDATION genera una sola candidata escrita a disco.
        # Si el archivo físico ya existe, se rechaza incondicionalmente para prevenir re-promociones.
        if self.json_path.exists() or self.sha_path.exists():
            raise MethodologicalContaminationError(
                "Promoción múltiple rechazada: VALIDATION ya ha consolidado una candidata oficial en disco."
            )
            
        cand = FrozenCandidate(candidate_id=candidate_id, parameters=parameters, validation_pf=val_pf)
        cand_dict = asdict(cand)
        
        # Guardar JSON a disco de forma atómica/canónica
        json_data = json.dumps(cand_dict, sort_keys=True, indent=2)
        self.json_path.write_text(json_data, encoding="utf-8")
        
        # Regla 3: El hash SHA256 se calcula desde el contenido canónico del JSON físico.
        external_hash = self._compute_file_sha256(self.json_path)
        self.sha_path.write_text(external_hash, encoding="utf-8")
        
        cand.hash_signature = external_hash
        self.frozen_candidate = cand
        return cand

    def load_frozen_candidate_for_test(self, cand: FrozenCandidate | None = None) -> FrozenCandidate:
        """
        Carga y certifica la inmutabilidad de la candidata leyendo de forma estricta
        el disco duro. Erradica la confianza en objetos pasados puramente por memoria.
        """
        # Regla 7 y 8: Si falta el JSON o el hash en disco, bloquear.
        if not self.json_path.exists():
            raise MethodologicalContaminationError(
                f"Archivo ausente: no se localiza el entregable físico obligatorio {self.json_path.name} en disco."
            )
        if not self.sha_path.exists():
            raise MethodologicalContaminationError(
                f"Firma ausente: no se localiza el archivo de verificación externa {self.sha_path.name} en disco."
            )
            
        # Regla 4 y 5: TEST carga desde disco y recalcula hash comparando con el archivo externo.
        actual_file_hash = self._compute_file_sha256(self.json_path)
        expected_hash = self.sha_path.read_text(encoding="utf-8").strip()
        
        # Regla 6: Si el hash no coincide, bloquear.
        if actual_file_hash != expected_hash:
            raise MethodologicalContaminationError(
                "Fraude o corrupción detectada: el contenido físico de FROZEN_CANDIDATE.json no coincide con su firma externa SHA256."
            )
            
        try:
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
            loaded_cand = FrozenCandidate(
                candidate_id=data["candidate_id"],
                parameters=data["parameters"],
                validation_pf=data["validation_pf"],
                hash_signature=actual_file_hash
            )
        except Exception as e:
            raise MethodologicalContaminationError(f"Estructura corrupta en FROZEN_CANDIDATE.json: {e}")
            
        self.frozen_candidate = loaded_cand
        return loaded_cand

    def authorize_test_evaluation(self, candidate_to_evaluate: FrozenCandidate) -> bool:
        """
        Verifica que la candidata bajo evaluación OOS corresponda exactamente
        a la versión en disco duro. Prohíbe autorizaciones de memoria volátil.
        """
        if self.current_phase != "TEST":
            raise MethodologicalContaminationError("Evaluación OOS reservada exclusivamente para la fase TEST.")
            
        # Forzar la carga física desde disco para asegurar que no se evada el chequeo
        disk_cand = self.load_frozen_candidate_for_test()
        
        # Regla 9 y 12: Bloquear si se pasa una config en memoria que difiere del disco o si mutó.
        # Comparamos tanto el candidate_id como los parámetros serializados.
        if candidate_to_evaluate.candidate_id != disk_cand.candidate_id:
            raise MethodologicalContaminationError(
                "Intento de sustitución de candidata bloqueado: el ID en memoria no coincide con el disco."
            )
            
        # Para ser absolutamente robustos ante mutaciones finas, serializamos y comparamos.
        mem_params = json.dumps(candidate_to_evaluate.parameters, sort_keys=True)
        disk_params = json.dumps(disk_cand.parameters, sort_keys=True)
        if mem_params != disk_params:
            raise MethodologicalContaminationError(
                "Intento de mutación paramétrica bloqueado: los diccionarios en memoria y disco difieren."
            )
            
        return True

    def emit_final_verdict(self, source_candidate: FrozenCandidate) -> dict[str, any]:
        """Emite el dictamen final validando unívocamente contra los registros del disco duro."""
        if self.current_phase == "TRAIN":
            raise MethodologicalContaminationError("Estrictamente prohibido emitir veredictos finales desde la fase TRAIN.")
        if self.current_phase != "TEST":
            raise MethodologicalContaminationError("El veredicto final requiere la previa superación de la fase TEST.")
            
        # Validación física por disco
        disk_cand = self.load_frozen_candidate_for_test()
        
        mem_params = json.dumps(source_candidate.parameters, sort_keys=True)
        disk_params = json.dumps(disk_cand.parameters, sort_keys=True)
        
        if source_candidate.candidate_id != disk_cand.candidate_id or mem_params != disk_params:
            raise MethodologicalContaminationError(
                "Fraude metodológico evitado: el dictamen final debe emanar exclusivamente de la candidata validada en disco."
            )
            
        return {
            "verdict": "AUTHORIZED_BY_WALKFORWARD_GUARD",
            "candidate_id": disk_cand.candidate_id,
            "signature": disk_cand.hash_signature
        }
