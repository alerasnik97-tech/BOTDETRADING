import hashlib, json, sys
from pathlib import Path

MANIFEST_PATH = Path(__file__).parent / "ENGINE_CORE_HASH_MANIFEST.json"
BASE_DIR = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB")
TARGET_DIRS = ["src/v6_utils", "src/v7_engine"]

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifiesto no encontrado en {MANIFEST_PATH}")
        sys.exit(1)
        
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    expected_files = {item["relative_path"]: item["sha256"] for item in manifest}
    actual_files = {}
    
    # Recorrer archivos físicos actuales
    for t_dir in TARGET_DIRS:
        root_path = BASE_DIR / t_dir
        if not root_path.exists():
            print(f"CRITICAL: Carpeta protegida base no hallada: {root_path}")
            sys.exit(1)
            
        for p in root_path.rglob("*.py"):
            rel_path = p.relative_to(BASE_DIR).as_posix()
            actual_files[rel_path] = compute_sha256(p)
            
    drift_detected = False
    missing_files = []
    modified_files = []
    new_files = []
    
    # Verificar faltantes y modificados
    for expected_path, expected_hash in expected_files.items():
        if expected_path not in actual_files:
            missing_files.append(expected_path)
            drift_detected = True
        else:
            if actual_files[expected_path] != expected_hash:
                modified_files.append((expected_path, expected_hash, actual_files[expected_path]))
                drift_detected = True
                
    # Verificar archivos nuevos no registrados en manifiesto
    for actual_path in actual_files:
        if actual_path not in expected_files:
            new_files.append(actual_path)
            drift_detected = True
            
    print("================================================================================")
    print("ENGINE CORE INTEGRITY VERIFICATION REPORT")
    print("================================================================================")
    print(f"Archivos esperados en manifiesto : {len(expected_files)}")
    print(f"Archivos físicos analizados      : {len(actual_files)}")
    print("--------------------------------------------------------------------------------")
    
    if missing_files:
        print("[!] ESTADO: ENGINE_CORE_MISSING_FILE")
        for f in missing_files: print(f"  - FALTANTE: {f}")
        
    if modified_files:
        print("[!] ESTADO: ENGINE_CORE_DRIFT_DETECTED")
        for f, h_exp, h_act in modified_files:
            print(f"  - MODIFICADO: {f}")
            print(f"    Esperado : {h_exp}")
            print(f"    Actual   : {h_act}")
            
    if new_files:
        print("[!] ESTADO: ENGINE_CORE_NEW_FILE_DETECTED")
        for f in new_files: print(f"  - INTRUSO (NUEVO): {f}")
        
    if drift_detected:
        print("--------------------------------------------------------------------------------")
        print("VEREDICTO: DRIFT CRÍTICO ENCONTRADO. EL MOTOR NO ES CONFIABLE.")
        print("================================================================================")
        sys.exit(1)
    else:
        print("[OK] ESTADO: ENGINE_CORE_OK")
        print("VEREDICTO: EL MOTOR CUMPLE 100% PARIDAD CANÓNICA INSTITUCIONAL.")
        print("================================================================================")
        sys.exit(0)

if __name__ == "__main__":
    main()
