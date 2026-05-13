# MANIFIESTO DE PRE-COMPROMISO METODOLÓGICO — R1 CANDIDATE FACTORY

## 1. Protocolo de Cegado de Prueba (Cero Test Leakage)
La fábrica de candidatos operará bajo un estricto régimen de walk-forward ciego:
- **Fase A (Exploración)**: Se utilizarán exclusivamente las particiones TRAIN (2020-22) y VAL (2023-24) para el escaneo amplio y la generación del ranking inicial.
- **Fase B (Selección)**: Los filtros de robustez se aplicarán sobre la muestra de validación y subperíodos históricos conocidos.
- **Fase C (Blind Single-Run)**: Solo los finalistas (Top 5) serán ejecutados sobre la partición TEST (2025-26). El resultado de TEST será definitivo; no se permitirán reajustes paramétricos tras la observación de esta partición.

## 2. Inmutabilidad de Parámetros
El espacio de búsqueda se congela en `R1_FACTORY_SEARCH_SPACE.json`. No se inyectarán nuevas variables de forma ad-hoc durante el proceso de filtrado.

## 3. Jerarquía de Métricas
La selección priorizará la **consistencia** (Sharpe/Expectancy) y la **resiliencia al slippage** por encima del retorno bruto absoluto.
