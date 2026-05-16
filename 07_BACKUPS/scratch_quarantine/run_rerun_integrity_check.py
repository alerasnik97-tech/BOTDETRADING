import subprocess
from pathlib import Path
import os
import json

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
ORCHESTRATOR = ROOT / "scratch" / "run_dual_line_daily_chain.py"

def test_rerun_protection():
    print("=== RERUN PROTECTION INTEGRITY TEST ===")
    test_date = "2026-04-22"
    lock_file = ROOT / "results" / f"chain_lock_{test_date}.json"
    
    # 1. Limpiar lock previo si existe
    if lock_file.exists():
        os.remove(lock_file)
        
    # 2. Primera ejecución (sin --run para no hacer nada real, solo el orquestador)
    print(f"Ejecutando primera vez para {test_date}...")
    res1 = subprocess.run(["python", str(ORCHESTRATOR), "--date", test_date], capture_output=True, text=True, cwd=str(ROOT))
    
    if lock_file.exists():
        print("[SUCCESS] Lockfile creado correctamente.")
    else:
        print("[FAILED] Lockfile no creado.")
        return

    # 3. Segunda ejecución (debería fallar por lockfile)
    print(f"Intentando re-ejecutar para {test_date}...")
    res2 = subprocess.run(["python", str(ORCHESTRATOR), "--date", test_date], capture_output=True, text=True, cwd=str(ROOT))
    
    if "FAIL-CLOSED" in res2.stderr or "FAIL-CLOSED" in res2.stdout:
        print("[SUCCESS] Rerun protection bloqueó la ejecución duplicada.")
    else:
        print("[FAILED] Rerun protection NO bloqueó la ejecución.")
        print(f"Stdout: {res2.stdout}")
        print(f"Stderr: {res2.stderr}")

if __name__ == "__main__":
    test_rerun_protection()
