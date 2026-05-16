# MANIPULANTE — Sistema de Trading Institucional
**Instrumento:** EURUSD | **Broker:** FTMO Demo (MetaTrader 5)
**Última auditoría:** 2026-05-06 | **Estado:** DEMO READY

---

## Qué es MANIPULANTE

Bot de trading institucional que opera barridos de liquidez (sweeps) 
en EURUSD durante la sesión de Nueva York. Detecta manipulación de 
precio en H1, confirma un cambio de carácter (CHOCH) en M3, y entra 
con gestión de riesgo fija.

---

## Parámetros sagrados (NO modificar sin auditoría)

| Parámetro          | Valor       |
|--------------------|-------------|
| Take Profit        | 1.4R        |
| Break Even trigger | 0.4R        |
| Body Filter        | ≥ 70%       |
| Ventana operativa  | 07:00–11:30 NY |
| Max trades/día     | 1           |
| Cierre viernes     | 16:55 NY    |
| Risk por trade     | 0.5%        |
| Modo actual        | DEMO ONLY   |

→ Fuente de autoridad: `01_ESTRATEGIA_AUTORIDAD\MANIPULANTE_DO_NOT_TOUCH.md`

---

## Resultados verificados (backtest 2015–2026)

| Métrica           | Valor       |
|-------------------|-------------|
| Trades            | 2,610       |
| Win Rate          | 32.5%       |
| Profit Factor     | 1.69        |
| Net R             | +420.6R     |
| Sharpe            | 2.99        |
| Max Drawdown      | −10.1R      |
| OOS 2022–2026     | PF 1.57     |

---

## Estructura de carpetas

- `01_ESTRATEGIA_AUTORIDAD/`: Documentos de especificación y configuración maestra.
- `02_AUDITORIA/`: Reportes forenses, validación histórica y logs de integridad.
- `03_OPERATIVA/`: Scripts de ejecución, conexión a MT5 y soporte de trials.
- `04_REPORTES/`: Métricas de rendimiento, curvas de equidad y análisis temporal.

---

## Inicio rápido

1. Verificar preflight: `03_OPERATIVA\PREFLIGHT_CHECKLIST.md`
2. Lanzar con el .bat correspondiente
3. Monitorear con el runbook: `03_OPERATIVA\FIRST_SESSION_RUNBOOK.md`

---

*Generado automáticamente por auditoría Phase64J — 2026-05-06*
