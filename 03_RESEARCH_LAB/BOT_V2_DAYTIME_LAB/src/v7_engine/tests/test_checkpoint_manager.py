import pytest
from pathlib import Path
from src.v7_engine.checkpoint_manager import CheckpointManager

def test_checkpoint_atomic_write(tmp_path):
    """1. El avance parcial se consolida en disco duro garantizando el reemplazo atómico."""
    mgr = CheckpointManager(worker_id="w1", base_dir=tmp_path)
    mgr.save_checkpoint("cfg_001", "CORE", "TRAIN", "COMPLETED", {"pf": 1.5}, "hashA")
    
    # Comprobar existencia
    assert mgr.checkpoint_file.exists()
    assert not mgr.checkpoint_file.with_suffix(".tmp").exists()
    
    content = mgr.checkpoint_file.read_text(encoding="utf-8")
    assert "cfg_001" in content
    assert "COMPLETED" in content
    assert "hashA" in content

def test_checkpoint_resume_skips_completed(tmp_path):
    """2. El escáner global aísla e identifica unívocamente las combinaciones exitosas."""
    mgr = CheckpointManager(worker_id="w1", base_dir=tmp_path)
    mgr.save_checkpoint("cfg_ok", "CORE", "TRAIN", "COMPLETED", {}, "h1")
    
    completed = mgr.get_completed_configs()
    assert "cfg_ok" in completed

def test_checkpoint_does_not_skip_failed(tmp_path):
    """3. Configuraciones marcadas como fallidas o interrumpidas jamás se asumen resueltas."""
    mgr = CheckpointManager(worker_id="w1", base_dir=tmp_path)
    mgr.save_checkpoint("cfg_err", "CORE", "TRAIN", "FAILED", {}, "h2")
    mgr.save_checkpoint("cfg_timeout", "CORE", "TRAIN", "INTERRUPTED", {}, "h3")
    
    completed = mgr.get_completed_configs()
    assert "cfg_err" not in completed
    assert "cfg_timeout" not in completed

def test_worker_checkpoints_do_not_collide(tmp_path):
    """4. Los archivos físicos de persistencia se separan de forma concurrente por identificador de worker."""
    m1 = CheckpointManager(worker_id="alpha", base_dir=tmp_path)
    m2 = CheckpointManager(worker_id="beta", base_dir=tmp_path)
    
    m1.save_checkpoint("c1", "EXP", "TRAIN", "COMPLETED", {}, "hA")
    m2.save_checkpoint("c2", "EXP", "TRAIN", "COMPLETED", {}, "hB")
    
    assert m1.checkpoint_file.name == "checkpoint_alpha.csv"
    assert m2.checkpoint_file.name == "checkpoint_beta.csv"
    
    # Ambos se consolidan correctamente a nivel de orquestación global
    assert m1.get_completed_configs() == {"c1", "c2"}

def test_checkpoint_contains_input_hash(tmp_path):
    """5. Se retiene la huella criptográfica de entrada paramétrica en el registro."""
    mgr = CheckpointManager(worker_id="w1", base_dir=tmp_path)
    mgr.save_checkpoint("c_hash", "CORE", "TRAIN", "COMPLETED", {}, "signature_xyz")
    
    text = mgr.checkpoint_file.read_text(encoding="utf-8")
    assert "signature_xyz" in text

def test_checkpoint_rejects_corrupt_file(tmp_path):
    """6. El arnés de reanudación descarta sectores corruptos sin bloquear la lectura global."""
    mgr = CheckpointManager(worker_id="bad", base_dir=tmp_path)
    # Escribimos basura binaria corrupta en el archivo
    mgr.checkpoint_file.write_bytes(b"\x00\xff\xfe\x00corrupted_bytes_without_headers")
    
    # La lectura tolera el fallo y retorna vacío sin provocar terminaciones abruptas
    assert mgr.get_completed_configs() == set()
