import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
docs = {
    '00_READ_THIS_FIRST.md': "\n- **Audit Status**: Phase 35 Final Readiness Audit Completed.\n- **Veredicto**: READY_FOR_MICRO_REAL_WITH_WARNINGS.\n- **Riesgo Inicial Real**: 0.10% a 0.25% únicamente.",
    '01_CURRENT_PROJECT_STATUS.md': "\n- **Phase Active**: PHASE35 Completed.\n- **Real Readiness**: Sí (Micro Real Manual).",
    '02_STRATEGY_AUTHORITY_MAP.md': "\n- **Status**: MANIPULANTE validada para paper y micro real manual.",
    'MANIPULANTE/00_LEER_PRIMERO/README_MANIPULANTE.md': "\n## Micro Real Readiness\nEl sistema ha pasado la auditoría Phase 35. Ver `MANIPULANTE/12_MICRO_REAL_READINESS/` para el plan de ejecución."
}

for rel_path, append_str in docs.items():
    p = ROOT / rel_path
    if p.exists():
        try:
            with open(p, 'a', encoding='utf-8') as f:
                f.write(append_str)
        except: pass

# Update status.json
s_path = ROOT / 'BOT_V2_DAYTIME_LAB' / 'status.json'
if s_path.exists():
    with open(s_path, 'r') as f:
        s = json.load(f)
    s['latest_phase_completed'] = "PHASE35"
    s['micro_real_readiness'] = "READY"
    with open(s_path, 'w') as f:
        json.dump(s, f, indent=2)

print("Master docs updated for Phase 35.")
