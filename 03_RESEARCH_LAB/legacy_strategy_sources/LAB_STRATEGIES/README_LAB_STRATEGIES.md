# LAB_STRATEGIES

`LAB_STRATEGIES/` es la zona amarilla del proyecto.

Objetivo: investigar nuevas familias de estrategias sin tocar MANIPULANTE.

Reglas:

- MANIPULANTE no se modifica desde esta zona.
- No se crean scripts live.
- No se abre MT5.
- No se envian, cierran ni modifican ordenes.
- No se suben secretos, tokens, `.env` ni data pesada.
- Toda estrategia nueva debe tener reporte, config aislada y evidencia reproducible.
- Toda comparacion contra MANIPULANTE ocurre despues de validar la estrategia sola.

Estructura inicial:

```text
LAB_STRATEGIES/
  README_LAB_STRATEGIES.md
  EURUSD_DAYTIME/
    README_EURUSD_DAYTIME_LAB.md
    strategies/
    shared/
    reports/
    correlation/
    _templates/
```

Verificacion recomendada antes de cerrar una sesion de research:

```powershell
python BOT_V2_DAYTIME_LAB/src/phase47a_lab_isolation_guard.py
git status --short
```
