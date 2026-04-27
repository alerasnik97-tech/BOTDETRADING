
import zipfile
import os
from pathlib import Path

def create_v2_zip():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB")
    zip_path = root / "000_PARA_CHATGPT.zip"
    
    exclude_dirs = {"__pycache__", ".git", "archive", "cache"}
    exclude_exts = {".pyc", ".tmp", ".log"}
    CSV_SIZE_LIMIT = 1024 * 1024 
    
    if zip_path.exists(): zip_path.unlink()
        
    print(f"Creating ZIP {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root_dir, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            for file in files:
                file_path = Path(root_dir) / file
                rel_path = file_path.relative_to(root)
                if rel_path.name == zip_path.name: continue
                if file_path.suffix.lower() in exclude_exts: continue
                if file_path.suffix.lower() == ".csv" and file_path.stat().st_size > CSV_SIZE_LIMIT: continue
                zipf.write(file_path, rel_path)
                
    print(f"ZIP created successfully.")

if __name__ == "__main__":
    create_v2_zip()


