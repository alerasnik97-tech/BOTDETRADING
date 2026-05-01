# PHASE48A — ORB V1 RESEARCH REPORT

## 1. Lo más importante
Se ha completado el primer backtest serio del laboratorio de investigación sobre la estrategia **01_ORB_V1 (Opening Range Breakout)** utilizando un dataset de alta precisión de 5 años (2020-2025). Los resultados muestran un **Profit Factor bruto de 1.063** para la mejor configuración encontrada (TP 1.0R). Este rendimiento es insuficiente para cubrir costos operativos (spread/slippage) y mantener robustez, por lo que la estrategia queda **RECHAZADA** en su forma actual.

## 2. Veredicto final exacto
**ORB_V1_REJECTED**

## 3. Worktree / Rama
- **Ruta**: `C:\Users\alera\Desktop\Bot\BOT_RESEARCH_WORKTREES\eurusd-daytime-strategy-01`
- **Branch**: `research/eurusd-daytime-strategy-01`
- **Status Git**: Limpio (solo archivos de investigación).

## 4. Dataset usado
- **Archivo**: `EURUSD_M1_BID_2020_2025.csv`
- **Timeframe**: M1
- **Rango fechas**: Enero 2020 – Diciembre 2025
- **Timezone**: UTC (procesado con offset NY UTC-5)
- **Calidad**: Alta precisión (Dukascopy).

## 5. Estrategia probada
- **Nombre**: 01_ORB_V1 — Opening Range Breakout
- **Reglas**:
  - Rango inicial: 07:00–08:00 NY.
  - Entrada: Ruptura confirmada por cierre de vela M1.
  - Filtro de Rango: 8 a 35 pips.
- **Parámetros evaluados**: TP 1.0R, 1.5R, 2.0R.

## 6. Resultados globales (Mejor Config: TP 1.0R)
- **Muestra**: 1011 trades (Muestra estadísticamente muy sólida).
- **PF Bruto**: **1.063**
- **Expectancy Bruta**: **0.03R**
- **Winrate**: ~51.5% (Estimado para TP 1.0R)
- **Estado**: **FALLIDO**. Un PF < 1.15 bruto no sobrevive a costos reales ni a la varianza del mercado.

## 7. Mejores configuraciones
1. **TP 1.0R**: PF 1.063 (Mejor estabilidad, pero edge nulo tras costos).
2. **TP 1.5R**: PF 0.886 (Pérdida neta).
3. **TP 2.0R**: PF 0.719 (Pérdida neta severa).

## 8. Robustez
- La estrategia muestra una degradación lineal a medida que aumenta el objetivo de profit, lo que indica que no hay un sesgo de continuación fuerte tras la ruptura del rango de apertura en EURUSD para estos horarios.

## 9. Correlación contra MANIPULANTE
- **Estado**: **MANIPULANTE_CORRELATION_PENDING**.
- **Nota**: No se procedió a la medición ya que la estrategia base no superó el gate de profit mínimo.

## 10. Riesgos / Limitaciones
- **Costos**: Tras aplicar spread de 0.8-1.2 pips, el PF de 1.06 pasaría a ser menor a 1.0.
- **Lógica**: El breakout simple de la primera hora de NY parece tener demasiada "ruido" o falsas rupturas sin filtros adicionales (como volumen o estructura superior).

## 11. Archivos generados
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/01_ORB_V1/orb_v1_results_summary.csv`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/01_ORB_V1/orb_v1_research_report.md`
- `LAB_STRATEGIES/EURUSD_DAYTIME/strategies/01_ORB_V1/orb_v1_research_report.json`

## 12. Seguridad
- **MANIPULANTE**: Intacto y protegido.
- **MT5/Órdenes**: No se realizaron conexiones ni ejecuciones.
- **Git**: No se realizó `git add .`, ni `commit`, ni `push`.

## 13. Siguiente paso único
**Proceder con la investigación de la estrategia 02_LCF_V1 (London Close Fade) en la cola de investigación.**
