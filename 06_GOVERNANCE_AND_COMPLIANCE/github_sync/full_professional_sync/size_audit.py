import os
import csv

root_dir = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
output_file = os.path.join(root_dir, "06_GOVERNANCE_AND_COMPLIANCE", "github_sync", "full_professional_sync", "GITHUB_SYNC_DATA_SIZE_AUDIT.csv")

results = []
for root, dirs, files in os.walk(root_dir):
    if ".git" in dirs:
        dirs.remove(".git")
    if "venv" in dirs:
        dirs.remove("venv")
    if ".venv" in dirs:
        dirs.remove(".venv")
        
    for file in files:
        file_path = os.path.join(root, file)
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb > 10:
                rel_path = os.path.relpath(file_path, root_dir)
                results.append({
                    "path": rel_path,
                    "size_mb": round(size_mb, 2),
                    "tracked_status": "Unknown",
                    "should_be_in_git": "NO",
                    "action": "Exclude"
                })
        except Exception:
            pass

with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["path", "size_mb", "tracked_status", "should_be_in_git", "action"])
    writer.writeheader()
    writer.writerows(results)
