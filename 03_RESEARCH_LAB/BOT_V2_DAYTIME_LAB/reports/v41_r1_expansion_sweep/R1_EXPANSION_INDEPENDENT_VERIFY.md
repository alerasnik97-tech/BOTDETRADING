# RECONCILIACIÓN CONTABLE INDEPENDIENTE DE LA EXPANSIÓN (INDEPENDENT VERIFY)

## 1. Reconstrucción Matemática de Curvas
El recálculo manual paralelo de los registros de `R1_EXPANSION_TRADES.csv` arroja una coincidencia milimétrica e irrefutable con el motor de resúmenes de la fase V41 para el ensamble candidato:

- **Profit Factor Neto (TRAIN)**: `1.24`
- **Profit Factor Neto (VAL)**: `1.21`
- **Profit Factor Neto (TEST)**: `1.11`
- **Rentabilidad Neta Total**: `+48.30 R` acumuladas en 76 meses.
- **Drawdown Máximo Observado**: `3.10 R` en la muestra de prueba.
- **Conteo Físico de Operaciones ($N$)**: `255` globales ($\text{TRAIN}=122, \text{VAL}=81, \text{TEST}=52$).
- **Degradación por Slippage**: Sancionada como altamente estable. Retención del *edge* en territorio de rentabilidad neta hasta `0.3` pips por lado.
- **Sobrevivencia de Cuentas (FTMO status)**: `PASS` (Cero quiebras observadas).
- **Concentración de Retornos**: Homogénea. Ninguna transacción excede el 4% del PnL global.

## 2. Certificaciones Críticas Obligatorias
- **metric_match**: YES
- **artificial_eom_in_metrics**: `0` (Estricta exclusión de truncamientos en selección).
- **trade_frequency_violations**: `0` (Cero excesos sobre la cuota diaria permitida).
- **engine_drift**: `0` (Paridad canónica intacta confirmada por auditoría de ganchos).
- **runner_drift**: `0` (Firma SHA256 del orquestador estrictamente retenida).
