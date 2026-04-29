
import os
from pathlib import Path

def recursive_zip_cleanup():
    root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
    canonical_name = "000_PARA_CHATGPT.zip"
    canonical_path = root / canonical_name
    
    print(f"Recursive ZIP Cleanup started from {root}...")
    
    count = 0
    for file_path in root.rglob('*.zip'):
        # Skip the canonical one in the root
        if file_path.absolute() == canonical_path.absolute():
            continue
            
        # Neutralize
        new_path = str(file_path) + "bak"
        try:
            os.rename(file_path, new_path)
            print(f"Neutralized: {file_path.relative_to(root)} -> {file_path.name}bak")
            count += 1
        except Exception as e:
            print(f"Error neutralizing {file_path}: {e}")
            
    print(f"Cleanup finished. Total neutralized: {count}")

if __name__ == "__main__":
    recursive_zip_cleanup()
