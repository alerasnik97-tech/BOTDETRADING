# PHASE 9 FREQUENCY EXPANSION REPORT

## 1. Objetivo
Evaluar la viabilidad de expandir la frecuencia operativa de la estrategia diurna EURUSD hacia los 10-20 trades/mes sin destruir el edge (PF >= 1.40).

## 2. Baseline High Precision (Preservada)
- **ID:** Phase 8 Candidate B
- **Sample:** 165 trades (con filtrado estricto)
- **PF:** 2.09
- **Frecuencia:** ~1.2 trades/mes

## 3. Diagnóstico de Frecuencia
Los filtros que más "matan" frecuencia son:
1. **Filtro de Cuerpo (60%):** Elimina el 39% de los setups.
2. **Exclusión de Viernes:** Elimina el 22% de la muestra.
3. **Selección de Niveles:** Usar solo PDH/L ignora setups de calidad en London High/Low.

## 4. Resultados de Relajación Individual
| Variante | Sample | Freq (m) | PF | Veredicto |
|----------|--------|----------|----|-----------|
| **Candidate B Baseline** | 237* | 1.8 | 1.92 | BASELINE_PLUS |
| **No Body Filter** | 271 | 2.0 | 1.64 | USEFUL_RELAXATION |
| **Include Fridays** | 305 | 2.3 | 1.64 | USEFUL_RELAXATION |
| **Expanded Window (14:00)** | 226 | 1.7 | 1.48 | MARGINAL |
| **Aggressive (All Relaxed)** | 493 | 3.7 | 1.36 | REJECTED_EDGE_LOSS |

*\*Nota: El incremento de 165 a 237 en la baseline se debe a la inclusión de niveles London/Asia en el motor.*

## 5. Laboratorio de Alta Frecuencia (Stress Test)
Se intentó alcanzar los 10-20 trades/mes mediante la eliminación de límites diarios y múltiples barridos.

- **Balanced_Candidate_M5:** Freq 21.6 trades/mes | PF **0.82** (FAIL)
- **Balanced_V2 (Selective):** Freq 8.8 trades/mes | PF **1.06** (FAIL)

## 6. Top Candidates Detectados
A pesar del fallo en frecuencia alta, se detectó un candidato equilibrado de frecuencia media:

### CANDIDATE_PHASE9_BALANCED (Recomendado)
- **Reglas:** PDH/L + London H/L + No Friday + Body 50%.
- **Frecuencia:** **2.8 trades/mes**.
- **Profit Factor:** **1.72**.
- **Expectancy:** +0.24 R.
- **Veredicto:** VALIDATED_MEDIUM_QUALITY.

## 7. Robustez 2023–2025
Al aumentar la frecuencia a 8-20 trades/mes, el sistema colapsa en casi todos los años, confirmando que la lógica de Mean Reversion diurna en EURUSD requiere **SELECTIVIDAD EXTREMA**.

## 8. Sensibilidad a Costos
Los candidatos de alta frecuencia (PF < 1.10) son destruidos instantáneamente por el spread (PF cae a < 0.80 con 0.5 pips de slippage). Solo el candidato de baja frecuencia sobrevive a la fricción.

## 9. Comparación Final
| Métrica | Phase 8 (High Prec) | Phase 9 (High Freq Attempt) | Resultado |
|---------|---------------------|-----------------------------|-----------|
| **Trades/Mes** | 1.2 | 21.6 | +1700% |
| **Profit Factor** | 2.09 | 0.82 | **-60% (Destrucción)** |

## 10. ¿Se llegó a 20 trades/mes?
**SÍ, pero el resultado fue una estrategia perdedora (PF 0.82).** No es posible operar esta lógica con esa frecuencia en el mercado real de EURUSD diurno.

## 11. Veredicto Final
**`PHASE9_ONLY_LOW_FREQUENCY_SURVIVES`**

La Phase 9 confirma científicamente que la estrategia **NO ESCALA** hacia alta frecuencia sin sacrificar toda la rentabilidad. Se recomienda mantener el enfoque de **ALTA PRECISIÓN** (Phase 8) con una frecuencia máxima de **2-3 trades al mes**.

## 12. Siguiente Paso Único
Aceptar la realidad matemática de la baja frecuencia y proceder al **Forward Testing** con el candidato Phase 8 + Filtros de niveles London validados.

---
*Reporte generado por el laboratorio de frecuencia Phase 9.*
