import os
import json

def test_protocol_integrity():
    base_path = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\micro_pilot_protocol"
    required_files = [
        "README.md",
        "MICRO_PILOT_PROTOCOL.md",
        "activation_checklist.md",
        "daily_operator_checklist.md",
        "kill_switch_rules.md",
        "risk_limits.md",
        "execution_rules.md",
        "escalation_rules.md",
        "status_template.json",
        "protocol_summary.md"
    ]
    
    forbidden_status = "NOT_ACTIVE_UNTIL_MICRO_PILOT_ALLOWED"
    
    print(f"--- Iniciando Validación de Integridad del Protocolo ---")
    
    all_ok = True
    
    # 1. Verificar existencia de archivos
    for f in required_files:
        path = os.path.join(base_path, f)
        if os.path.exists(path):
            print(f"[OK] Archivo presente: {f}")
        else:
            print(f"[FAIL] Archivo ausente: {f}")
            all_ok = False
            
    # 2. Verificar estado explícito en MDs
    for f in required_files:
        if f.endswith(".md"):
            path = os.path.join(base_path, f)
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                if forbidden_status in content:
                    print(f"[OK] Estado bloqueado confirmado en: {f}")
                else:
                    print(f"[FAIL] Falta estado bloqueado en: {f}")
                    all_ok = False
                    
    # 3. Verificar status_template.json
    template_path = os.path.join(base_path, "status_template.json")
    if os.path.exists(template_path):
        with open(template_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if data.get("allowed_to_trade") is False and data.get("activation_status") == forbidden_status:
                print(f"[OK] status_template.json arranca bloqueado correctamente.")
            else:
                print(f"[FAIL] status_template.json tiene valores de activacion incorrectos.")
                all_ok = False

    if all_ok:
        print(f"\nRESULTADO FINAL: CONSISTENCIA TOTAL CONFIRMADA.")
    else:
        print(f"\nRESULTADO FINAL: SE DETECTARON FALLOS DE CONSISTENCIA.")
        
if __name__ == "__main__":
    test_protocol_integrity()
