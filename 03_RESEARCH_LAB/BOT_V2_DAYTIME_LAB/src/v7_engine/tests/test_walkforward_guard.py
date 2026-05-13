import json
import pytest
from pathlib import Path
from src.v7_engine.walkforward_guard import WalkForwardGuard, FrozenCandidate, MethodologicalContaminationError

def test_test_requires_physical_frozen_candidate_json(tmp_path):
    """1. TEST bloquea la evaluación si no se localiza el archivo físico FROZEN_CANDIDATE.json."""
    guard = WalkForwardGuard(current_phase="TEST", storage_dir=tmp_path)
    # sha256 file exists but json is missing
    (tmp_path / "FROZEN_CANDIDATE_SHA256.txt").write_text("dummyhash", encoding="utf-8")
    cand = FrozenCandidate("c1", {}, 1.0)
    with pytest.raises(MethodologicalContaminationError) as exc:
        guard.authorize_test_evaluation(cand)
    assert "FROZEN_CANDIDATE.json" in str(exc.value)

def test_test_requires_external_sha256_file(tmp_path):
    """2. TEST bloquea si falta el archivo externo FROZEN_CANDIDATE_SHA256.txt."""
    guard = WalkForwardGuard(current_phase="TEST", storage_dir=tmp_path)
    (tmp_path / "FROZEN_CANDIDATE.json").write_text('{"candidate_id":"c1"}', encoding="utf-8")
    cand = FrozenCandidate("c1", {}, 1.0)
    with pytest.raises(MethodologicalContaminationError) as exc:
        guard.authorize_test_evaluation(cand)
    assert "FROZEN_CANDIDATE_SHA256.txt" in str(exc.value)

def test_test_rejects_memory_only_candidate(tmp_path):
    """3. TEST rechaza autorizaciones basadas exclusivamente en memoria sin contraparte en disco."""
    guard = WalkForwardGuard(current_phase="TEST", storage_dir=tmp_path)
    cand = FrozenCandidate("mem_only", {"x": 1}, 1.5)
    with pytest.raises(MethodologicalContaminationError):
        guard.authorize_test_evaluation(cand)

def test_test_rejects_hash_mismatch(tmp_path):
    """4. TEST bloquea incondicionalmente si el SHA256 del archivo difiere de la firma guardada."""
    guard = WalkForwardGuard(current_phase="TEST", storage_dir=tmp_path)
    json_path = tmp_path / "FROZEN_CANDIDATE.json"
    sha_path = tmp_path / "FROZEN_CANDIDATE_SHA256.txt"
    
    json_path.write_text('{"candidate_id":"alpha","parameters":{}}', encoding="utf-8")
    sha_path.write_text("invalid_manipulated_hash_string", encoding="utf-8")
    
    cand = FrozenCandidate("alpha", {}, 1.0)
    with pytest.raises(MethodologicalContaminationError) as exc:
        guard.authorize_test_evaluation(cand)
    assert "no coincide con su firma externa" in str(exc.value)

def test_test_rejects_mutated_candidate_file(tmp_path):
    """5. TEST detecta si se mutó el contenido físico o si se le pasa una config distinta en memoria."""
    # Primero congelamos de forma válida
    guard_val = WalkForwardGuard(current_phase="VALIDATION", storage_dir=tmp_path)
    cand_official = guard_val.select_frozen_candidate("official", {"sl": 20}, 1.40)
    
    # Transicionamos a TEST
    guard_test = WalkForwardGuard(current_phase="TEST", storage_dir=tmp_path)
    
    # Intentamos evaluar con un objeto mutado en memoria
    cand_mutated = FrozenCandidate("official", {"sl": 999}, 1.40)
    with pytest.raises(MethodologicalContaminationError) as exc:
        guard_test.authorize_test_evaluation(cand_mutated)
    assert "diccionarios en memoria y disco difieren" in str(exc.value)

def test_test_rejects_missing_frozen_candidate(tmp_path):
    """6. TEST rechaza evaluación de prueba si falta la candidata oficial."""
    guard = WalkForwardGuard(current_phase="TEST", storage_dir=tmp_path)
    cand = FrozenCandidate("missing", {}, 1.0)
    with pytest.raises(MethodologicalContaminationError):
        guard.authorize_test_evaluation(cand)

def test_test_rejects_missing_sha_file(tmp_path):
    """7. TEST rechaza evaluación si falta el archivo del hash externo."""
    guard = WalkForwardGuard(current_phase="TEST", storage_dir=tmp_path)
    (tmp_path / "FROZEN_CANDIDATE.json").write_text('{"candidate_id":"ok"}', encoding="utf-8")
    cand = FrozenCandidate("ok", {}, 1.0)
    with pytest.raises(MethodologicalContaminationError):
        guard.authorize_test_evaluation(cand)

def test_test_allows_only_disk_verified_candidate(tmp_path):
    """8. TEST autoriza exclusivamente cuando la candidata bajo evaluación coincide 100% con disco."""
    guard_val = WalkForwardGuard(current_phase="VALIDATION", storage_dir=tmp_path)
    official = guard_val.select_frozen_candidate("deployable", {"tp": 2.5, "be": 1.2}, 1.55)
    
    guard_test = WalkForwardGuard(current_phase="TEST", storage_dir=tmp_path)
    # Se le pasa el objeto idéntico en memoria, y contra el disco coincide a la perfección.
    assert guard_test.authorize_test_evaluation(official) is True
    
    # Veredicto final emitido exitosamente
    res = guard_test.emit_final_verdict(official)
    assert res["verdict"] == "AUTHORIZED_BY_WALKFORWARD_GUARD"
    assert res["candidate_id"] == "deployable"

def test_validation_writes_single_frozen_candidate(tmp_path):
    """9. VALIDATION escribe a disco el JSON y su hash unívocamente."""
    guard = WalkForwardGuard(current_phase="VALIDATION", storage_dir=tmp_path)
    cand = guard.select_frozen_candidate("base_win", {"x": 10}, 1.30)
    
    assert (tmp_path / "FROZEN_CANDIDATE.json").exists()
    assert (tmp_path / "FROZEN_CANDIDATE_SHA256.txt").exists()
    
    disk_hash = (tmp_path / "FROZEN_CANDIDATE_SHA256.txt").read_text(encoding="utf-8")
    assert len(disk_hash.strip()) == 64
    assert cand.hash_signature == disk_hash.strip()

def test_validation_rejects_second_frozen_candidate(tmp_path):
    """10. VALIDATION rechaza la promoción de una segunda candidata si los archivos ya existen."""
    guard = WalkForwardGuard(current_phase="VALIDATION", storage_dir=tmp_path)
    guard.select_frozen_candidate("cand_one", {"v": 1}, 1.20)
    
    # Intento secundario
    with pytest.raises(MethodologicalContaminationError) as exc:
        guard.select_frozen_candidate("cand_two", {"v": 2}, 1.50)
    assert "ya ha consolidado una candidata oficial en disco" in str(exc.value)

def test_test_rejects_ranking_operation():
    """11. TEST bloquea incondicionalmente operaciones de ranking y optimización fina."""
    guard = WalkForwardGuard(current_phase="TEST")
    with pytest.raises(MethodologicalContaminationError):
        guard.authorize_ranking(10)

def test_test_rejects_search_space_iteration():
    """12. TEST rechaza consultas o iteraciones sobre el espacio de búsqueda."""
    guard = WalkForwardGuard(current_phase="TEST")
    with pytest.raises(MethodologicalContaminationError):
        guard.authorize_search_space_iteration()
