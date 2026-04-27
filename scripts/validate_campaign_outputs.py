import os
import sys
from pathlib import Path

# Configuración Canónica
CANONICAL_ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo").resolve()

# Requisitos del contrato de outputs
MANDATORY_OUTPUTS = [
    "CAMPAIGN_INTAKE",
    "CAMPAIGN_FINAL_DECISION",
    "CAMPAIGN_LEDGER",
    "CAMPAIGN_OUTPUT_CONTRACT.md"
]

def main():
    print("="*60)
    print("CAMPAIGN OUTPUT VALIDATOR")
    print("="*60)
    
    # Esta es una validación genérica. En uso real, se le pasaría el ID de campaña.
    if len(sys.argv) < 2:
        print("\nSTATUS: WARN")
        print("  No se especificó ID de campaña para validación profunda.")
        print("  Verificando existencia del sistema base...")
        
        missing = []
        for item in ["CAMPAIGN_GATEKEEPER_PROTOCOL.md", "CAMPAIGN_OUTPUT_CONTRACT.md", "RESEARCH_DECISION_MATRIX.md"]:
            if not (CANONICAL_ROOT / item).exists():
                missing.append(item)
        
        if missing:
            print(f"  [ERROR] Faltan piezas del sistema canonico: {missing}")
            sys.exit(1)
        
        print("  Sistema base: PASS")
        sys.exit(0)

    campaign_id = sys.argv[1]
    errors = []
    
    # Simulación de búsqueda de archivos con el ID
    files_in_root = [f.name for f in CANONICAL_ROOT.iterdir() if f.is_file()]
    
    for req in MANDATORY_OUTPUTS:
        if req.endswith(".md") or req.endswith(".csv"):
             if not (CANONICAL_ROOT / req).exists():
                 errors.append(f"Archivo obligatorio inexistente: {req}")
        else:
            # Búsqueda por prefijo/patrón
            found = any(campaign_id in f and req in f for f in files_in_root)
            if not found:
                errors.append(f"No se encontró el output '{req}' para la campaña '{campaign_id}'")

    if errors:
        print(f"\nSTATUS: FAIL-CLOSED (Campaign: {campaign_id})")
        for err in errors:
            print(f"  [ERROR] {err}")
        sys.exit(1)
        
    print(f"\nSTATUS: PASS (Campaign: {campaign_id})")
    print("  Todos los artefactos del contrato han sido detectados.")
    sys.exit(0)

if __name__ == "__main__":
    main()
