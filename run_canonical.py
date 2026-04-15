import os
import sys
import subprocess
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("ERROR: Uso incorrecto.")
        print("USO CANONICO: python run_canonical.py <nombre_estrategia> [modo: normal|stress|precision]")
        print("Ejemplo: python run_canonical.py ny_br_ema stress")
        sys.exit(1)
        
    strategy_name = sys.argv[1]
    
    # 1. HARDENING: Validar nombre de estrategia contra el registro
    root_dir = Path(__file__).parent
    sys.path.insert(0, str(root_dir))
    
    try:
        from research_lab.config import STRATEGY_NAMES
        if strategy_name not in STRATEGY_NAMES:
            print(f"ERROR CANONICO: La estrategia '{strategy_name}' no existe en STRATEGY_NAMES.")
            print(f"Estrategias registradas validas: {', '.join(STRATEGY_NAMES)}")
            sys.exit(1)
    except Exception as e:
        print(f"WARN CANONICO: No se pudo verificar el registro de estrategias ({e}). Continuum ciego.")

    # 2. HARDENING: Modo de ejecución estricto
    execution_mode = "normal" # Default inseguro forzado
    if len(sys.argv) >= 3:
        mode_arg = sys.argv[2]
        if mode_arg in ["normal", "stress", "precision"]:
            execution_mode = mode_arg
        else:
            print(f"ERROR CANONICO: Modo de ejecución '{mode_arg}' es inválido.")
            print("Modos aceptados para corridas serias: normal, stress, precision")
            sys.exit(1)

    # 3. HARDENING: Deteccion de argumentos inesperados (preveniendo over-riding hacky)
    if len(sys.argv) > 3:
        print(f"ERROR CANONICO: Argumentos excesivos u opciones sueltas detectadas: {sys.argv[3:]}")
        print("El wrapper oficial no permite hackear comandos (ej. start_date). Usa el engine directamente si quieres debuggear, pero NO generaras un lineage valido.")
        sys.exit(1)
        
    main_py = root_dir / "research_lab" / "main.py"
    
    cmd = [
        sys.executable, str(main_py),
        "run", "--strategy", strategy_name,
        "--end", "2025-12-31", # Forzar siempre el fin máximo de muestra
        "--execution-mode", execution_mode
    ]
    
    print("="*60)
    print(f"  EJECUCION CANONICA: {strategy_name}")
    print(f"  MODO DE ENGINE : {execution_mode.upper()}")
    print("="*60)
    
    if execution_mode == "normal":
        print(">>> WARNING: Ejecutando en NORMAL. Propenso a fills optimistas intradiarios. No usar para tomar decision final.")
    
    try:
        subprocess.run(cmd, cwd=root_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR CANONICO: Corrida abortada por rechazo matematico (harness) o fallo del motor interno. Codigo: {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nCANCELADO POR EL USUARIO.")
        sys.exit(1)

if __name__ == "__main__":
    main()
