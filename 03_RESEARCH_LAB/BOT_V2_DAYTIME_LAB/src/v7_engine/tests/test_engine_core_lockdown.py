import json, os, subprocess, sys
from pathlib import Path
import pytest

VERIFY_SCRIPT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\06_GOVERNANCE_AND_COMPLIANCE\engine_lockdown\ENGINE_CORE_VERIFY.py")
MANIFEST_PATH = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\06_GOVERNANCE_AND_COMPLIANCE\engine_lockdown\ENGINE_CORE_HASH_MANIFEST.json")
PYTHON_EXE = sys.executable

def test_manifest_contains_required_namespaces():
    assert MANIFEST_PATH.exists(), "El manifiesto institucional no existe"
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    has_v7 = any(item["relative_path"].startswith("src/v7_engine") for item in manifest)
    has_v6 = any(item["relative_path"].startswith("src/v6_utils") for item in manifest)
    
    assert has_v7, "El manifiesto no incluye archivos protegidos de v7_engine"
    assert has_v6, "El manifiesto no incluye archivos protegidos de v6_utils"
    assert len(manifest) >= 70, "El conteo de archivos en el manifiesto es inusualmente bajo"

def test_engine_core_verify_passes_clean_working_tree():
    """Verifica que el script pase nativamente con el working tree canónico actual"""
    res = subprocess.run([PYTHON_EXE, str(VERIFY_SCRIPT)], capture_output=True, text=True)
    assert res.returncode == 0, f"El script de verificación falló inesperadamente:\n{res.stdout}\n{res.stderr}"
    assert "[OK] ESTADO: ENGINE_CORE_OK" in res.stdout

def test_verify_script_logic_simulations(tmp_path):
    """
    Simula de forma segura los 3 modos de falla (drift, missing, intruder)
    importando dinámicamente el módulo de verificación y aislando sus rutas
    en un sandbox temporal.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("engine_verify", str(VERIFY_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 1. Crear estructura sandbox en tmp_path
    sandbox_base = tmp_path / "sandbox"
    (sandbox_base / "src/v6_utils").mkdir(parents=True)
    (sandbox_base / "src/v7_engine").mkdir(parents=True)
    
    f1 = sandbox_base / "src/v7_engine/engine.py"
    f1.write_text("print('core inmutable')")
    h1 = module.compute_sha256(f1)
    
    f2 = sandbox_base / "src/v6_utils/bars.py"
    f2.write_text("print('bars inmutable')")
    h2 = module.compute_sha256(f2)
    
    dummy_manifest = [
        {"relative_path": "src/v7_engine/engine.py", "sha256": h1},
        {"relative_path": "src/v6_utils/bars.py", "sha256": h2}
    ]
    
    sandbox_manifest = tmp_path / "ENGINE_CORE_HASH_MANIFEST.json"
    sandbox_manifest.write_text(json.dumps(dummy_manifest))
    
    # Parchear variables en el módulo cargado
    module.MANIFEST_PATH = sandbox_manifest
    module.BASE_DIR = sandbox_base
    
    # Caso A: Paridad limpia en el sandbox
    with pytest.raises(SystemExit) as e:
        module.main()
    assert e.value.code == 0
    
    # Caso B: Drift (hash incorrecto)
    f1.write_text("print('drift introducido')")
    with pytest.raises(SystemExit) as e:
        module.main()
    assert e.value.code == 1
    
    # Restaurar
    f1.write_text("print('core inmutable')")
    
    # Caso C: Missing file
    f2.unlink()
    with pytest.raises(SystemExit) as e:
        module.main()
    assert e.value.code == 1
    
    # Restaurar
    f2.write_text("print('bars inmutable')")
    
    # Caso D: Intruso (archivo nuevo)
    intruder = sandbox_base / "src/v7_engine/hack.py"
    intruder.write_text("print('intruso')")
    with pytest.raises(SystemExit) as e:
        module.main()
    assert e.value.code == 1
