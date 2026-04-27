# Política del Micro Piloto Real (MPR)
## Nivel de Gobernanza: ULTRA-CONSERVADOR

Esta política define las condiciones bajo las cuales una variante en incubación shadow puede ser habilitada para operar con capital real en una escala mínima ("Micro Piloto").

---

## 1. Naturaleza del Micro Piloto
- **Propósito:** Validar la ejecución técnica y el comportamiento emocional/operativo en entorno real.
- **Capital:** Riesgo ultra-bajo (Micro-lotes).
- **Alcance:** Una sola variante autorizada.
- **Duración:** Mínimo N=20 trades reales antes de cualquier revisión de escalado.

## 2. Gates de Habilitación
### A. Robustez de Investigación (Research)
- El candidato debe estar en estado `SHADOW_READY`.
- No debe haber discrepancias en la lógica auditada.
- Los reportes de `axis_scan` deben respaldar la variante seleccionada.

### B. Gobernanza Shadow
- `Shadow Autopilot` debe reportar estado `OK`.
- `Checkpoint Review System` debe estar activo.
- La bitácora operativa acumulada debe estar libre de errores técnicos recurrentes.

### C. Evidencia Mínima (Forward Shadow)
- Al menos N=10 trades shadow con comportamiento coherente.
- El tribunal de evidencia no debe tener alertas críticas (`SHADOW_HOLD`).
- Drawdown shadow controlado (< 10R).

### D. Contención de Riesgo
- Protocolo de riesgo del micro piloto aceptado.
- Kill switch definido y operativo.
- Checklist de activación completada.

---
## 3. Filosofía
Preferimos detener un piloto sano que dejar correr uno enfermo. La seguridad del capital es la prioridad absoluta.
