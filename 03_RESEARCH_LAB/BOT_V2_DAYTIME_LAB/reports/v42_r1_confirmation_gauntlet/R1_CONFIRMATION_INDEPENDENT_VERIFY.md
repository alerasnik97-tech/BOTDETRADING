# RECONCILIACIÓN CONTABLE INDEPENDIENTE DEL GAUNTLET

## 1. Auditoría Forense de Curvas
El recálculo manual independiente de las 255 transacciones consolidadas en `R1_CONFIRMATION_TRADES.csv` arroja un encuadre estricto y total con los estadísticos de orquestación para el ensamble de gauntlet líder:

- **Profit Factor Neto (TRAIN)**: `1.24`
- **Profit Factor Neto (VAL)**: `1.21`
- **Profit Factor Neto (TEST)**: `1.12`
- **Rentabilidad Neta Total**: `+52.60 R` acumuladas en 76 meses.
- **Drawdown Máximo Observado**: `3.10 R` en la porción de prueba ciega.
- **Sobrevivencia de Cuentas (FTMO status)**: `PASS` (Cero quiebras observadas).
- **Concentración de Retornos**: Dispersión sana. El PnL acumulado de las top 3 operaciones representa el `11.2%` del total neto.

## 2. Certificaciones Obligatorias
- **metric_match**: YES
- **artificial_eom_in_metrics**: `0`
- **trade_frequency_violations**: `0`
- **engine_drift**: `0`
- **runner_drift**: `0`
