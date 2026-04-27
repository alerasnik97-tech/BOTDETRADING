
import sys

def check_imports():
    required = ['pandas', 'numpy', 'pytz', 'MetaTrader5', 'dateutil']
    missing = []
    for lib in required:
        try:
            __import__(lib)
            print(f"[OK] {lib} importado.")
        except ImportError:
            missing.append(lib)
    
    if missing:
        print(f"[ERROR] Faltan librerías: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("[SUCCESS] Todas las dependencias críticas están presentes.")

if __name__ == "__main__":
    check_imports()
