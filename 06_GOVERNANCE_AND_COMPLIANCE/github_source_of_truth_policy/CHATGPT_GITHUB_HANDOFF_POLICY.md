# CHATGPT GITHUB HANDOFF POLICY

## 1. Formato Obligatorio de Entrega
Cada finalización de tarea o fase debe concluir con un reporte siguiendo esta estructura:

---
## Estado
[READY / BLOCKED / RED / INCONCLUSIVE]

## GitHub
- **repo**: alerasnik97-tech/bottrading
- **branch**: [rama canónica vigente declarada en el gate — actualmente `governance/root-hygiene-20260516`; NUNCA hardcodear `clean-sync-branch`]
- **commit**: [Hash del último commit]
- **push**: SUCCESS (non-force)
- **main touched**: NO
- **force push**: NO
- **ZIP used**: NO
- **blockers**: [lista de blockers vigentes o NONE]

## Archivos clave
- [Ruta del archivo 1]
- [Ruta del archivo 2]
- [Ruta del archivo 3]

## Evidencia de Validación
- **tests**: [Passed/Failed]
- **engine verify**: [OK/FAIL]
- **rowcount**: [Match/Mismatch]
- **metric match**: [YES/NO]
- **no TEST leakage**: [CONFIRMED]

## Prohibiciones respetadas
- [x] no data mutation
- [x] no engine touched
- [x] no push to main
- [x] no ZIP used as workflow
- [x] no Explorer opened
---

## 2. Instrucciones para la IA
La IA receptora debe clonar o actualizar el repositorio localmente en la rama indicada para proceder con la revisión de la evidencia física.
