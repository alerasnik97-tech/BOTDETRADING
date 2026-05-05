import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
docs = {
    '00_READ_THIS_FIRST.md': "\n- **Ruta Oficial Única**: `C:\\Users\\alera\\Desktop\\Bot\\BOT DE TRADING ultimo`\n- **Ejecución Manual**: La ejecución actual es manual y temporal. El operador humano NO TIENE discrecionalidad estratégica. Solo ejecuta la estrategia 100% programada (MANIPULANTE).",
    '01_CURRENT_PROJECT_STATUS.md': "\n- **Ruta Oficial Única**: Consolidada en `C:\\Users\\alera\\Desktop\\Bot\\BOT DE TRADING ultimo`.\n- **Fase de Ejecución**: Manual Temporal (Sin discrecionalidad estratégica).",
    '02_STRATEGY_AUTHORITY_MAP.md': "\n- **Aclaración**: El trader actúa como operador de registro. MANIPULANTE es 100% programable.",
    'ESTRUCTURA_DEL_PROYECTO.md': "\n## Ruta Oficial Única\nTodo el proyecto debe residir exclusivamente en `C:\\Users\\alera\\Desktop\\Bot\\BOT DE TRADING ultimo`.\n\n## Core Protocols\nLos documentos base canónicos de arquitectura se encuentran unificados en `BOT_V2_DAYTIME_LAB/docs/CORE_PROTOCOLS/`.",
    'ABRIR_MANIPULANTE_AQUI.txt': "\nNOTA: La fase manual es temporal. Usted NO tiene permiso para alterar las reglas. Siga las señales de forma 100% objetiva.",
    'ZIP_CONTENTS_MANIFEST.md': "\n- Contiene `CORE_PROTOCOLS` unificados.\n- Contiene clarificación de ejecución manual (`MANUAL_EXECUTION_BOUNDARY.md`).\n- Reporte Phase34.",
    'BOT_V2_DAYTIME_LAB/ZIP_CONTENTS_MANIFEST.md': "\n- `docs/CORE_PROTOCOLS/` (Fuente canónica).\n- `docs/MANUAL_EXECUTION_BOUNDARY.md`.\n- Outputs livianos de Phase34."
}

for rel_path, append_str in docs.items():
    p = ROOT / rel_path
    if p.exists():
        try:
            with open(p, 'r', encoding='utf-8') as f:
                content = f.read()
            if "Ruta Oficial Única" not in content and "fase manual es temporal" not in content and "CORE_PROTOCOLS" not in content:
                with open(p, 'a', encoding='utf-8') as f:
                    f.write(append_str)
        except Exception:
            pass

# JSONs
j1 = ROOT / '01_CURRENT_PROJECT_STATUS.json'
if j1.exists():
    try:
        with open(j1, 'r') as f:
            d = json.load(f)
        d['official_path'] = str(ROOT)
        d['manual_execution'] = "Temporal_No_Discretion"
        with open(j1, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception: pass

j2 = ROOT / '02_STRATEGY_AUTHORITY_MAP.json'
if j2.exists():
    try:
        with open(j2, 'r') as f:
            d = json.load(f)
        d['manual_execution_rule'] = "100_percent_programmable_no_human_edge"
        with open(j2, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception: pass

j3 = ROOT / 'BOT_V2_DAYTIME_LAB' / 'status.json'
if j3.exists():
    try:
        with open(j3, 'r') as f:
            d = json.load(f)
        d['latest_phase_active'] = "PHASE34"
        d['canonical_path_sync'] = True
        with open(j3, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception: pass

print("Master docs updated.")
