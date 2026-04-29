import os
import sys
import json
from pathlib import Path
from datetime import datetime

# ======================================================================
# MANIPULANTE — QUICK STATUS PANEL (PHASE 37ZE)
# ======================================================================

ROOT = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
LOG_DIR = ROOT / "MANIPULANTE" / "10_LOGS_PAPER" / "ftmo_trial_bot"
QUICK_STATUS = LOG_DIR / "quick_status.txt"
HEARTBEAT = LOG_DIR / "heartbeat.json"

def get_runner_count():
    try:
        import subprocess
        cmd = 'powershell -NoProfile -ExecutionPolicy Bypass -Command "(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like \'*phase37_ftmo_trial_bot_runner.py*\' }).Count"'
        res = subprocess.check_output(cmd, shell=True).decode().strip()
        return int(res) if res else 0
    except:
        return 0

def get_mt5_status():
    try:
        import subprocess
        res = subprocess.check_output('tasklist /FI "IMAGENAME eq terminal64.exe"', shell=True).decode()
        return "ABIERTO" if "terminal64.exe" in res else "CERRADO"
    except:
        return "DESCONOCIDO"

def read_quick_status():
    data = {}
    if QUICK_STATUS.exists():
        with open(QUICK_STATUS, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    k, v = line.strip().split("=", 1)
                    data[k] = v
    return data

def render():
    runners = get_runner_count()
    mt5 = get_mt5_status()
    qs = read_quick_status()
    
    estado_final = "ROJO"
    mensaje = "BOT APAGADO"
    color_code = "\033[91m" # Red
    
    if runners > 1:
        estado_final = "VIOLETA"
        mensaje = "REVISAR DUPLICADOS"
        color_code = "\033[95m" # Purple
    elif runners == 1:
        estado_final = qs.get("ESTADO_GENERAL", "VERDE")
        mensaje = qs.get("MENSAJE", "BOT ACTIVO")
        
        if estado_final == "VERDE": color_code = "\033[92m"
        elif estado_final == "AMARILLO": color_code = "\033[93m"
        elif estado_final == "CRITICO": color_code = "\033[41m\033[37m" # White on Red
    
    reset = "\033[0m"
    
    print("="*70)
    print(" MANIPULANTE — PANEL DE ESTADO")
    print(f" Actualiza cada 30 segundos | PID Runner: {runners}")
    print("="*70)
    print()
    print(f" ESTADO GENERAL: {color_code} {mensaje} {reset}")
    print(f" CUENTA:         {qs.get('CUENTA', 'DESCONOCIDA')}")
    print(f" RUNNER:         {'ACTIVO' if runners > 0 else 'APAGADO'}")
    print(f" MT5:            {mt5}")
    print()
    print(f" NEWS:           {qs.get('NEWS', '---')}")
    print(f" ULTIMA DECISION: {qs.get('ULTIMA_DECISION', '---')}")
    print(f" OPERACION ABIERTA: {qs.get('OPERACION_ABIERTA', '---')}")
    
    safe_off = qs.get('SEGURO_APAGAR_PC', 'NO')
    safe_color = "\033[92m" if safe_off == "SI" else "\033[91m"
    print(f" SEGURO APAGAR PC: {safe_color} {safe_off} {reset}")
    print()
    print(f" ULTIMA ACTUALIZACION: {qs.get('ULTIMA_ACTUALIZACION_ARG', '---')} ARG / {qs.get('ULTIMA_ACTUALIZACION_NY', '---')} NY")
    print()
    print("="*70)
    print(" SIGNIFICADO")
    print(" [VERDE] Todo bien")
    print(" [AMARILLO] Bot activo pero no opera por regla (Noticia/Horario)")
    print(" [ROJO] Bot apagado o error")
    print(" [CRITICO] No apagar PC")
    print(" [VIOLETA] Revisar duplicados")
    print("="*70)
    print()
    print(" Ctrl+C para cerrar este panel. El bot NO se apaga.")
    print("="*70)

if __name__ == "__main__":
    # Windows ANSI support
    if os.name == 'nt':
        os.system('')
    render()
