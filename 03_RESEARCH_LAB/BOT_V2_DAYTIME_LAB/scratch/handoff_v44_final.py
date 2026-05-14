import os
import shutil
import hashlib
import zipfile
from pathlib import Path

def get_file_metadata(filepath: Path):
    if not filepath.exists():
        return None
    size_mb = filepath.stat().st_size / (1024 * 1024)
    sha256 = hashlib.sha256(filepath.read_bytes()).hexdigest()
    
    testzip_result = None
    names = []
    with zipfile.ZipFile(filepath, "r") as zf:
        testzip_result = zf.testzip()
        names = zf.namelist()
        
    total_files = len(names)
    contains_v44 = any("v44_r1_final_candidate_confirmation" in n for n in names)
    contains_decision = any("R1_FINAL_CONFIRMATION_DECISION" in n for n in names)
    contains_results = any("R1_FINAL_CONFIRMATION_RESULTS_TEST" in n for n in names)
    contains_claude = any("claude_47_audit_pack" in n for n in names)
    
    return {
        "size_mb": size_mb,
        "sha256": sha256,
        "testzip": testzip_result,
        "total_files": total_files,
        "contains_v44": contains_v44,
        "contains_decision": contains_decision,
        "contains_results": contains_results,
        "contains_claude": contains_claude
    }

def execute_handoff():
    root_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    official_zip = root_dir / "000_PARA_CHATGPT.zip"
    upload_dir = Path(r"C:\Users\alera\Desktop\UPLOAD_CHATGPT_ACTUAL")
    upload_zip = upload_dir / "CHATGPT_UPLOAD_R1_V44_FINAL_b76d23cdde5a.zip"
    
    print("1. VERIFICANDO ZIP OFICIAL LOCAL...")
    off_meta = get_file_metadata(official_zip)
    if not off_meta:
        print("ERROR: El archivo ZIP oficial no existe.")
        return
    
    expected_hash = "b76d23cdde5a96f05d536a4778b5ad0a6ed3a343105d565b3e4eed54d2c83dc3"
    if off_meta["sha256"] != expected_hash:
        print(f"ADVERTENCIA: El hash oficial {off_meta['sha256']} no coincide con {expected_hash}.")
    else:
        print(f"Hash oficial certificado! ({expected_hash})")

    print("2. LIMPIANDO CARPETA DE SUBIDA EXTERNA...")
    upload_dir.mkdir(parents=True, exist_ok=True)
    for f in upload_dir.iterdir():
        if f.suffix == ".zip":
            print(f"Borrando ZIP viejo: {f.name}...")
            f.unlink()
            
    print(f"3. CREANDO COPIA DE SUBIDA: {upload_zip.name}...")
    shutil.copy2(official_zip, upload_zip)
    
    print("4. AUDITORA FINAL DE LA COPIA...")
    up_meta = get_file_metadata(upload_zip)
    
    same_hash = off_meta["sha256"] == up_meta["sha256"]
    
    print(f"\n--- REPORTE DE HANDOFF V44 ---")
    print(f"Copia creada: {upload_zip}")
    print(f"Same hash: {'YES' if same_hash else 'NO'}")
    print(f"Total files: {up_meta['total_files']} (Esperado: 804)")
    print(f"Contains V44: {'YES' if up_meta['contains_v44'] else 'NO'}")
    print(f"Contains Decision: {'YES' if up_meta['contains_decision'] else 'NO'}")
    print(f"Contains Results: {'YES' if up_meta['contains_results'] else 'NO'}")
    print(f"Contains Claude Pack: {'YES' if up_meta['contains_claude'] else 'NO'}")
    print(f"Testzip: {up_meta['testzip']}")
    print(f"------------------------------\n")

if __name__ == "__main__":
    execute_handoff()
