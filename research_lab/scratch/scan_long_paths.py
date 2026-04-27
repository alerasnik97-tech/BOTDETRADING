import os

def scan_long_paths(root_dir, threshold=240):
    long_paths = []
    print(f"Escaneando {root_dir}...")
    # Use extended path prefix for the root to be safe
    root_dir_extended = "\\\\?\\" + os.path.abspath(root_dir)
    
    for root, dirs, files in os.walk(root_dir_extended):
        for item in dirs + files:
            full_path = os.path.join(root, item)
            # Remove the prefix for length comparison to match user's perspective
            display_path = full_path.replace("\\\\?\\", "")
            if len(display_path) > threshold:
                long_paths.append(display_path)
    
    return long_paths

if __name__ == "__main__":
    target = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
    findings = scan_long_paths(target)
    
    if findings:
        print(f"\n[!] Se encontraron {len(findings)} rutas que exceden el límite:")
        for p in findings:
            print(p)
    else:
        print("\n[OK] No se encontraron rutas excesivamente largas. El árbol es saludable.")
