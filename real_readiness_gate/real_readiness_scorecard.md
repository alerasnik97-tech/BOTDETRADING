# Real Readiness Scorecard
## Reporte de Evaluación Institucional

**Línea Evaluada:** `tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m`
**Fecha de Evaluación:** `2026-04-24T00:07:16.414714Z`
**Veredicto:** `SHADOW_READY`

---

## 1. RESUMEN DE GATES

| Gate | Valor | Umbral | Estado |
|------|-------|---------|--------|
| Sample size | 1638 | 1500 | ✅ PASS |
| Profit factor | 2.4741 | 2.0 | ✅ PASS |
| Expectancy | 0.486 | 0.3 | ✅ PASS |
| Max drawdown | 11.0 | 15.0 | ✅ PASS |
| Year positive ratio | 1.0 | 1.0 | ✅ PASS |
| Year concentration | 0.1845 | 0.3 | ✅ PASS |
| Operational namespace | MISSING | N/A | ❌ FAIL |
| Timeout dependency | 0.1972 | 0.4 | ✅ PASS |
| Shadow execution | 0 | 20 | ❌ FAIL |

---

## 2. BLOQUEOS IDENTIFICADOS

- `MISSING_OPERATIONAL_NAMESPACE`
- `INSUFFICIENT_SHADOW_SAMPLE`

---

## 3. SIGUIENTE PASO ÚNICO

**ACTIVAR INFRAESTRUCTURA SHADOW.** La línea es robusta en backtest. Se requiere crear el namespace `results/shadow` e iniciar ejecución forward controlada (N=20).