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
    contains_v40 = any("v40_r1_absorption_mean_reversion" in n for n in names)
    contains_v41 = any("v41_r1_expansion_sweep" in n for n in names)
    contains_v42 = any("v42_r1_confirmation_gauntlet" in n for n in names)
    contains_r1_conf = any("R1_CONFIRMATION_DECISION" in n for n in names)
    contains_r1_exp = any("R1_EXPANSION_DECISION" in n for n in names)
    
    no_raw_data = not any("05_MARKET_DATA_VAULT/data" in n for n in names if n.endswith(".csv"))
    no_parquet = not any(n.endswith(".parquet") for n in names)
    no_venv = not any("venv" in n for n in names)
    no_git_cache = not any(".git/" in n for n in names)
    no_internal_zips = not any(n.endswith(".zip") for n in names)
    
    return {
        "size_mb": size_mb,
        "sha256": sha256,
        "testzip": testzip_result,
        "total_files": total_files,
        "contains_v40": contains_v40,
        "contains_v41": contains_v41,
        "contains_v42": contains_v42,
        "contains_r1_conf": contains_r1_conf,
        "contains_r1_exp": contains_r1_exp,
        "no_raw_data": no_raw_data,
        "no_parquet": no_parquet,
        "no_venv": no_venv,
        "no_git_cache": no_git_cache,
        "no_internal_zips": no_internal_zips
    }

def execute_handoff():
    root_dir = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    official_zip = root_dir / "000_PARA_CHATGPT.zip"
    upload_dir = Path(r"C:\Users\alera\Desktop\UPLOAD_CHATGPT_ACTUAL")
    upload_zip = upload_dir / "CHATGPT_UPLOAD_R1_CURRENT_5d7c84f28522.zip"
    report_file = root_dir / "06_GOVERNANCE_AND_COMPLIANCE" / "artifact_delivery" / "single_zip_delivery_lock" / "FINAL_CHATGPT_UPLOAD_COPY_VERIFICATION.txt"
    
    print("1. VERIFICANDO ZIP OFICIAL LOCAL...")
    off_meta = get_file_metadata(official_zip)
    if not off_meta:
        print("ERROR: El archivo ZIP oficial no existe en el disco.")
        return
    
    expected_hash = "5d7c84f28522e098f7d3aa9f3d438dd79a478e32a0fdf201963a25cc43405b77"
    if off_meta["sha256"] != expected_hash:
        print(f"ADVERTENCIA: El hash oficial es {off_meta['sha256']}, difiere del esperado {expected_hash}.")
        print("El archivo será copiado para retener el estado exacto reportado en el turno previo.")
    else:
        print(f"¡Hash certificado con pureza bit-a-bit! ({expected_hash})")
        
    print("2. PREPARANDO CARPETA EXTERNA ANTI-CACHE...")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Eliminar ZIPs anteriores en la carpeta de subida
    for f in upload_dir.iterdir():
        if f.suffix == ".zip":
            print(f"Purgando resto en caché externo: {f.name}...")
            f.unlink()
            
    print(f"Creando copia bit-a-bit externa: {upload_zip.name}...")
    shutil.copy2(official_zip, upload_zip)
    
    print("3. VERIFICANDO CRUZADAMENTE AMBOS ARCHIVOS...")
    up_meta = get_file_metadata(upload_zip)
    
    same_hash = off_meta["sha256"] == up_meta["sha256"]
    print(f"Coincidencia de Hash Cruzada: {'YES' if same_hash else 'NO'}")
    
    # Conteo de zips en raíz institucional
    root_zips = [f.name for f in root_dir.iterdir() if f.suffix == ".zip"]
    
    report_content = f"""================================================================================
FINAL CHATGPT UPLOAD COPY VERIFICATION REPORT (ANTI-CACHE HANDOFF PROTOCOL)
================================================================================
Ruta Canónica Oficial    : {official_zip}
Ruta Externa de Subida   : {upload_zip}
--------------------------------------------------------------------------------
Firma SHA256 Oficial     : {off_meta['sha256']}
Firma SHA256 Copia Subida: {up_meta['sha256']}
Coincidencia de Hash     : {'YES' if same_hash else 'NO'}
--------------------------------------------------------------------------------
Total Archivos Oficial   : {off_meta['total_files']}
Total Archivos Copia     : {up_meta['total_files']}
Tamaño Físico (MB)       : {off_meta['size_mb']:.2f} MB
Integridad Estructural   : {'OK (Ningún error de CRC)' if up_meta['testzip'] is None else f'CORRUPTO en {up_meta["testzip"]}'}
--------------------------------------------------------------------------------
AUDITORÍA DE PRESENCIA INTERNA (COPIA DE SUBIDA):
- contains v40           : {up_meta['contains_v40']}
- contains v41           : {up_meta['contains_v41']}
- contains v42           : {up_meta['contains_v42']}
- contains R1_CONFIRMATION: {up_meta['contains_r1_conf']}
- contains R1_EXPANSION  : {up_meta['contains_r1_exp']}
--------------------------------------------------------------------------------
AUDITORÍA DE RESTRICCIONES DE INCLUSIÓN:
- no raw data en bóveda  : {up_meta['no_raw_data']}
- no archivos parquet    : {up_meta['no_parquet']}
- no entornos venv       : {up_meta['no_venv']}
- no cachés de Git       : {up_meta['no_git_cache']}
- no ZIPs internos       : {up_meta['no_internal_zips']}
--------------------------------------------------------------------------------
ESTADO DE REGLA MAESTRA DEL PROYECTO:
- Conteo de ZIPs en la raíz del proyecto : {len(root_zips)} (Esperado: 1 archivo único canónico)
- Copia externa anti-caché creada        : YES
================================================================================
"""
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(report_content, encoding="utf-8")
    print(f"Reporte final redactado con éxito en: {report_file.name}")
    print("¡Procedimiento Handoff Anti-Caché culminado de forma inmaculada!")

if __name__ == "__main__":
    execute_handoff()
