# AUDITORÍA DE SELECCIÓN V49 — R1

## 1. Proceso de Selección
- Universo Total: 100 (B1) + 300 (B2) + 100 (B3) = 500 configuraciones reales.
- Filtro de Robustez: Mínimo 20 trades totales en TRAIN+VAL.
- Ranking: Ordenado por PF_val descendente, luego PF_train.

## 2. Resultados Top 5
Los finalistas muestran estabilidad entre TRAIN y VAL, con PFs superiores a 1.10 en la mayoría de los casos.

## 3. Prevención de Overfitting
- No se utilizó el periodo TEST (2025-2026) en ninguna etapa.
- La selección se basa en la consistencia entre dos periodos independientes (TRAIN y VAL).
- Se incluyó análisis de estrés por slippage (0.3 y 0.5) para descartar estrategias hipersensibles.
