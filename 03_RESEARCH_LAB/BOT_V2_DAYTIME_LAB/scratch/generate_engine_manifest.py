import csv, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\BOT_V2_DAYTIME_LAB")
TARGET_DIRS = ["src/v6_utils", "src/v7_engine"]
OUT_DIR = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\06_GOVERNANCE_AND_COMPLIANCE\engine_lockdown")

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_list = []
    
    for t_dir in TARGET_DIRS:
        root_path = BASE_DIR / t_dir
        for p in root_path.rglob("*.py"):
            rel_path = p.relative_to(BASE_DIR).as_posix()
            sha = compute_sha256(p)
            stat = p.stat()
            dt_utc = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            
            manifest_list.append({
                "relative_path": rel_path,
                "sha256": sha,
                "size_bytes": stat.st_size,
                "last_modified_utc": dt_utc,
                "protected": "true",
                "status": "CANONICAL_GIT_RESTORED"
            })
            
    # Ordenar alfabéticamente para reproducibilidad
    manifest_list.sort(key=lambda x: x["relative_path"])
    
    # Escribir CSV
    csv_path = OUT_DIR / "ENGINE_CORE_HASH_MANIFEST.csv"
    fields = ["relative_path", "sha256", "size_bytes", "last_modified_utc", "protected", "status"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest_list)
        
    # Escribir JSON
    json_path = OUT_DIR / "ENGINE_CORE_HASH_MANIFEST.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_list, f, indent=2)
        
    print(f"Manifest generado exitosamente: {len(manifest_list)} archivos protegidos registrados.")

if __name__ == "__main__":
    main()
