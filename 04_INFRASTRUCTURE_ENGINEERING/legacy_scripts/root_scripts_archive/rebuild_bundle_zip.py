
import zipfile
import os
from pathlib import Path

def rebuild_zip():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    zip_path = root / "000_PARA_CHATGPT.zip"
    
    # Basic exclusions (will be refined in loop)
    exclude_dirs_names = {
        "data_precision",
        "data_precision_raw",
        "data_candidates_2022_2025",
        "data_free_2020",
        "__pycache__",
        ".git",
        ".venv",
        ".vscode"
    }
    
    exclude_exts = {".csv", ".gz", ".bi5", ".pyc", ".tmp", ".log", ".zip"}
    
    print(f"Rebuilding {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root_dir, dirs, files in os.walk(root):
            # Prune excluded directory names
            dirs[:] = [d for d in dirs if d not in exclude_dirs_names and not d.startswith("backup")]
            
            for file in files:
                file_path = Path(root_dir) / file
                rel_path = file_path.relative_to(root)
                
                # Exclude specific extensions
                if file_path.suffix.lower() in exclude_exts:
                    continue
                
                # Logic for data_intake: only include md/json/py
                if "data_intake" in str(rel_path):
                    if file_path.suffix.lower() not in [".md", ".json", ".py"]:
                        continue
                
                # Include mostly scripts and documents
                if file_path.suffix.lower() in [".md", ".json", ".py", ".ps1", ".bat"]:
                    zipf.write(file_path, rel_path)
                    
    print("ZIP rebuild complete.")

if __name__ == "__main__":
    rebuild_zip()
