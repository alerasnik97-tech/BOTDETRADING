# MANIPULANTE COST AUDIT FTMO REPORT

## 1. Lo mas importante
La estrategia **MANIPULANTE (Phase 25)** ha sido sometida a una auditoria de costos realistas para una cuenta **FTMO 10k Swing**. El veredicto es altamente positivo: el edge de la estrategia es lo suficientemente robusto como para absorber no solo las comisiones estandar de FTMO, sino tambien escenarios de stress de hasta 10 USD/lote y deslizamientos adicionales.

## 2. Veredicto final exacto
**COST_AUDIT_PASS_ROBUST**

## 3. Costos en auditoria original
- **Spread**: No incluido explicitamente como variable (dato basado en ejecucion de backtest). Se agrego stress de 0.5 pips adicionales en escenario I.
- **Comision**: No incluida (0R).
- **Swap**: No incluido. Se agrego stress en escenario J.
- **Slippage**: No incluido.

## 4. Supuestos usados
- **Cuenta**: 10,000 USD.
- **Riesgo**: 0.50% (50 USD por trade).
- **Lotaje**: Estimado dinamicamente segun la distancia al SL (promedio ~0.15 - 0.20 lots).
- **Escenarios**: 5, 7 y 10 USD/lote round-turn.

## 5. Tabla Comparativa de Escenarios

| Escenario | PF | Exp R | Max DD | Total R | Meses Neg | Veredicto |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BASELINE** | 2.79 | 0.28R | -5.58R | 737.5R | 8 | BASE |
| **FTMO 5 USD/lot** | 2.24 | 0.23R | -6.65R | 610.7R | 10 | **PASS** |
| **REF 7 USD/lot** | 2.07 | 0.21R | -7.20R | 560.0R | 15 | **PASS** |
| **STRESS 10 USD/lot** | 1.84 | 0.18R | -8.04R | 484.0R | 22 | **PASS** |
| **SPREAD STRESS (+0.5p)**| 1.84 | 0.18R | -8.04R | 484.0R | 22 | **PASS** |
| **FIXED 0.05R** | 2.21 | 0.23R | -6.43R | 606.2R | 12 | **PASS** |

## 6. Impacto en Break-Even (BE)
- **Total BE**: 1,329 trades.
- **Impacto**: El 100% de los BE brutos pasan a ser pequeñas perdidas netas.
- **Perdida Acumulada por BE (5 USD/lot)**: **-65.61R**.
- **Conclusion**: A pesar de que los BE ahora restan, el PF se mantiene por encima de 2.0 en escenarios realistas. No se recomienda ajustar el offset del BE (actualmente 0.4) hasta tener mas data de forward trading.

## 7. Impacto en Forced Close
- **Total FC**: 100 trades.
- **R Bruto**: +11.07R.
- **R Neto (5 USD/lot)**: +8.80R.
- **Conclusion**: Los cierres forzados siguen siendo rentables en el agregado, incluso despues de costos.

## 8. FTMO 10k Simulation (Riesgo 0.50%)
- **Resultado Neto Estimado (5 USD/lot)**: +30,536 USD (desde 2015).
- **Drawdown Maximo USD**: -332 USD (6.65R).
- **Comision Acumulada**: ~6,337 USD.
- **Resultado 2025**: +28.34R (~1,417 USD netos).

## 9. ¿El edge sobrevive costos?
**SI**. La estrategia mantiene un Profit Factor superior a 2.0 y una esperanza matematica superior a 0.20R en el escenario estandar de FTMO.

## 10. Recomendacion de operacion
- **Demo**: Seguir operando en Demo/Trial para validar el slippage real de FTMO.
- **Cuenta Paga**: La estrategia es tecnicamente apta para una evaluacion real de 10k. El bajo drawdown relativo permite una gestion de riesgo tranquila.

## 11. Limitaciones
- La estimacion de lotaje asume un pip value fijo (EURUSD 10 USD/pip/lot).
- El swap se aplico como un stress fijo de -2 USD/lot en trades overnight.
- No se considera el costo de la plataforma ni del VPS.
