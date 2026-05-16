# Shadow Line Lab
## Infraestructura de Incubación Forward Aislada

**Estado:** `RESEARCH_ONLY` / `SHADOW_ONLY`
**Candidato:** `tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m`

---

## 1. ¿Qué es este Lab?
Es un entorno de ejecución espejo, totalmente aislado del core productivo del bot. Su propósito es ejecutar el candidato seleccionado en modo shadow para recolectar métricas forward reales sin contaminar los ledgers oficiales ni interferir con la línea principal.

## 2. ¿Qué NO es?
- No es una línea de producción.
- No es un generador de señales para trading real.
- No modifica el autopilot ni los validadores del sistema base.

## 3. Arquitectura
- `runner_shadow.py`: Implementación fiel de la lógica del candidato.
- `orchestrator.py`: Controlador de ejecución diaria.
- `ledger_io.py`: Capa de persistencia aislada.
- `results/`: Directorio de salidas (ledger, status, telemetría).
- `outputs/`: Reportes de ejecución.

## 4. Cómo Ejecutar
Para correr una sesión shadow manual:
```powershell
python shadow_line_lab/orchestrator.py
```

## 5. Salidas Generadas
- `shadow_line_lab/results/shadow_ledger.csv`: Registro de trades (paper).
- `shadow_line_lab/results/shadow_daily_status.json`: Estado de la última ejecución.
- `shadow_line_lab/results/shadow_telemetry.log`: Trazabilidad técnica.
- `shadow_line_lab/outputs/shadow_run_report.md`: Reporte ejecutivo.

---
**SEGURIDAD:** Este módulo tiene prohibido escribir fuera de `shadow_line_lab/`. Los datasets de entrada se consumen en modo solo lectura.
