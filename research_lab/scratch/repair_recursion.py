import os
import shutil
import stat

def remove_readonly(func, path, exc_info):
    """Clear the readonly bit and re-attempt the deletion."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def repair():
    # Use the extended path prefix to handle more than 260 characters
    path_root = r"\\?\C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\legacy_archive_2026"
    print(f"Scanning {path_root} for recursion...")
    
    # Walk bottom-up to handle the deepest nested folders first
    for root, dirs, files in os.walk(path_root, topdown=False):
        for d in dirs:
            if "PARA CHATGPT" in d:
                full_path = os.path.join(root, d)
                print(f"Removing: {full_path}")
                try:
                    shutil.rmtree(full_path, onexc=remove_readonly)
                except Exception as e:
                    print(f"Failed to remove {full_path}: {e}")

if __name__ == "__main__":
    repair()
