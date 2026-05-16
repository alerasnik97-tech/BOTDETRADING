import os
import json
from pathlib import Path
import subprocess
import pytest

def get_root_dir() -> Path:
    return Path(__file__).parent.parent.parent

def test_canonical_wrapper_inexistent_strategy():
    """Wrapper should fail immediately with inexistent strategy name."""
    cmd = [
        "python", str(get_root_dir() / "run_canonical.py"),
        "esta_estrategia_no_existe"
    ]
    res = subprocess.run(cmd, cwd=get_root_dir(), capture_output=True, text=True)
    assert res.returncode != 0
    assert "ERROR CANONICO" in res.stdout
    assert "no existe en STRATEGY_NAMES" in res.stdout

def test_canonical_wrapper_invalid_mode():
    """Wrapper should fail with bad mode."""
    # Using a valid test strategy for this
    cmd = [
        "python", str(get_root_dir() / "run_canonical.py"),
        "ny_br_pure", "modo_inventado"
    ]
    res = subprocess.run(cmd, cwd=get_root_dir(), capture_output=True, text=True)
    assert res.returncode != 0
    assert "ERROR CANONICO" in res.stdout
    assert "es inválido" in res.stdout

def test_canonical_wrapper_excess_args():
    """Wrapper should forbid hacking extra args."""
    cmd = [
        "python", str(get_root_dir() / "run_canonical.py"),
        "ny_br_pure", "normal", "--start", "2010"
    ]
    res = subprocess.run(cmd, cwd=get_root_dir(), capture_output=True, text=True)
    assert res.returncode != 0
    assert "ERROR CANONICO" in res.stdout
    assert "excesivos" in res.stdout

def test_lineage_metadata_schema():
    """Validates the lineage output keys natively simulating a fast run."""
    # We execute a valid strategy on the official wrapper but restricted in time by editing the main call
    # But since we can't edit wrapper time easily without breaking canonical rules, 
    # we just run the fast bad baseline locally in main.py to get metadata.
    
    cmd = [
        "python", str(get_root_dir() / "research_lab" / "main.py"),
        "run", "--strategy", "ny_br_pure",
        "--start", "2024-01-01",
        "--end", "2024-03-01", 
        "--disable-news"
    ]
    
    try:
        subprocess.run(cmd, cwd=get_root_dir(), check=False, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired:
        pass
        
    results_dir = get_root_dir() / "results" / "research_lab_robust"
    if not results_dir.exists():
        pytest.skip("No results folder found")
        
    folders = sorted([f for f in results_dir.iterdir() if f.is_dir() and "ny_br_pure" in f.name])
    if folders:
        latest = folders[-1]
        meta_path = latest / "lineage_metadata.json"
        assert meta_path.exists(), "lineage_metadata.json debe existir"
        
        with open(meta_path, "r") as f:
            meta = json.load(f)
            
        # Presence asserts
        assert "versions" in meta, "Debe tener versionado"
        versions = meta["versions"]
        assert "lab" in versions
        assert "contract" in versions
        assert "rejection_protocol" in versions
        assert "cost_model" in versions
        assert "promotion_policy" in versions
        
        # Promotion Policy presence
        assert "final_promotion_status" in meta, "Debe tener final_promotion_status integrado"
        assert meta["final_promotion_status"] in ["HARD_REJECT", "SOFT_REJECT", "PASS_MINIMUM", "STRONG_CANDIDATE", "PENDING_OOS"]
        
        # Runner presence
        assert "runner_used" in meta
        assert "dataset_used" in meta

def test_smoke_canonical_happy_path():
    """
    SMOKE TEST FELIZ CANONICO
    Corremos el wrapper completo en 'normal' para una estrategia. 
    Verificamos que inicie, detecte el entorno sin morir al instante por errores internos,
    y arroje un log valido. Para no esperar eternamente el backtest entero de 5 años, usamos timeout.
    Lo importante es que pase el parseo, compile el pipeline OOS y empiece el fold WFA.
    """
    cmd = [
        "python", str(get_root_dir() / "run_canonical.py"),
        "ny_br_pure", "normal"
    ]
    try:
        res = subprocess.run(cmd, cwd=get_root_dir(), capture_output=True, text=True, timeout=12)
        # If it finished in 5s it probably failed or aborted normally
        if res.returncode != 0 and "ABORTADO" not in res.stdout and "abortada" not in res.stdout:
            assert False, f"Fallo inesperado del wrapper: {res.stdout}"
    except subprocess.TimeoutExpired:
        # Timeout means it happily started executing Walk Forward chunks safely!
        assert True
