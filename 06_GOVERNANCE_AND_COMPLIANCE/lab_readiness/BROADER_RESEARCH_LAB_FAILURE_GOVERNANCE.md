# BROADER RESEARCH LAB FAILURE GOVERNANCE

## 1. Status
**RED_WITH_DEFERRED_MODULES**

## 2. Audit of Failures (2026-05-16)
La suite general de `research_lab` presenta fallos en los siguientes módulos:

### A. High-Precision Engine (`test_level3_precision.py`)
- **Status**: **DEFERRED / BLOCKED FOR PRODUCTION**
- **Reason**: Discrepancias en el cálculo de slippage forzado y tiempos de salida.
- **Impact**: Bloquea cualquier investigación que requiera certificación de "Alta Precisión" (Dukascopy M1 Bid/Ask).
- **Train Lab Impact**: No bloqueante para investigación OHLCV M5 base.

### B. News Module (`test_news_filter.py`, `test_am_news_builder.py`)
- **Status**: **DEFERRED / BLOCKED FOR RESEARCH**
- **Reason**: Los archivos de noticias legacy fueron purgados o están deshabilitados.
- **Impact**: Bloquea estrategias que dependen de filtros de noticias reales.
- **Train Lab Impact**: No bloqueante siempre que las estrategias se prueben con `news_enabled=False`.

### C. Legacy USDJPY / Gold
- **Status**: **DEFERRED**
- **Reason**: El foco actual es EURUSD. Los tests de otros símbolos no han sido actualizados a la nueva arquitectura 2026.
- **Impact**: Ninguno para el scope actual.

## 3. Governance Decision
Se autoriza la apertura del laboratorio para **EURUSD TRAIN-ONLY OHLCV MODE** bajo las siguientes condiciones:
1. El motor en modo normal (`normal_mode`) está certificado por `test_engine.py` (PASS).
2. La infraestructura de datos EURUSD está certificada (97/100).
3. Los fallos de alta precisión y noticias quedan registrados como "Módulos Diferidos" y deben ser resueltos antes de pasar a fases de validación o forward-testing que requieran dicha precisión.

## 4. Maintenance Plan
- Las regresiones de `level3_precision` deben ser atendidas en el primer sprint post-apertura.
- El módulo de noticias será reconstruido íntegramente en la Fase E.
