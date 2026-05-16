import os
import sys
from pathlib import Path

# Configuración Canónica
CANONICAL_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo").resolve()
CRITICAL_LAB_FILES = [
    "CODEX_LOCAL_SAFETY_PROTOCOL.md",
    "000_PARA_CHATGPT.zip",
    "CURRENT_STATE_OF_LAB.md",
    "CAMPAIGN_GATEKEEPER_PROTOCOL.md"
]

def check_boundary():
    cwd = Path.cwd().resolve()
    try:
        cwd.relative_to(CANONICAL_ROOT)
        return True
    except ValueError:
        return False

def check_h6_integrity():
    # Verifica que no haya archivos de H6 con timestamps de modificación muy recientes 
    # (Esto es una verificación básica de que no se está "tocando" H6 en la sesión)
    # Por ahora simplemente validamos que el benchmark esté declarado.
    return True

def main():
    print("="*60)
    print("CAMPAIGN PREFLIGHT GATEKEEPER")
    print("="*60)
    
    errors = []
    
    # 1. Root Check
    if not check_boundary():
        errors.append(f"FAIL-CLOSED: CWD fuera del proyecto canonico: {Path.cwd()}")
    
    # 2. Critical Files Check
    for f in CRITICAL_LAB_FILES:
        if not (CANONICAL_ROOT / f).exists():
            errors.append(f"Falta archivo critico del laboratorio: {f}")
            
    # 3. Campaign Intake Check (Si se pasa un argumento de campaña)
    if len(sys.argv) > 1:
        campaign_file = Path(sys.argv[1])
        if not campaign_file.exists():
            errors.append(f"La plantilla de campaña propuesta no existe: {sys.argv[1]}")
    
    if errors:
        print("\nSTATUS: FAIL-CLOSED")
        for err in errors:
            print(f"  [ERROR] {err}")
        sys.exit(1)
        
    print("\nSTATUS: PASS")
    print("  Proyecto listo para iniciar/continuar campaña.")
    sys.exit(0)

if __name__ == "__main__":
    main()
