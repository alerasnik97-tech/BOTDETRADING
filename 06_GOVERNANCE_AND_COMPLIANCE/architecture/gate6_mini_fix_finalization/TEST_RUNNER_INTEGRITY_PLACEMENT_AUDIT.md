# AUDITORÍA DE UBICACIÓN E INTEGRACIÓN DE LA SUITE DE PRUEBAS DEL RUNNER
**Archivo de Pruebas:** `test_runner_integrity.py`  
**Fase:** Gate 6 Mini Fix Finalization  
**Fecha:** 2026-05-13  
**Estado de Integración:** `SUITE_INTEGRATION_VERIFIED_SUCCESS`

---

## 1. Trazabilidad de Migración de Archivos
*   **Ruta Origen (Evidencia Documental Previa):** `06_GOVERNANCE_AND_COMPLIANCE/architecture/gate6_mini_fix_runner_integrity/test_runner_integrity.py`
*   **Ruta Destino Final (Laboratorio Real):** `03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB/src/v7_engine/tests/test_runner_integrity.py`
*   **Importado Nativamente por Pytest:** **YES**

La consolidación del archivo directamente en el directorio de aserciones del motor unificado asegura que estas pruebas de regresión formen parte obligatoria de cada evaluación en integración continua.

## 2. Cobertura de Pruebas Detectada
El archivo contiene un total de **6 aserciones unitarias discretas** (`test_count_detected = 6`), diseñadas para evaluar rigurosamente el endurecimiento causal del backtest:

### A. Validación Causal V2_B (No-Clon)
*   **Función:** `test_gate6_mini_v2b_stop_is_not_market_clone`
*   **Objetivo:** Certifica que la orden Stop exige un cruce físico verificado sobre el vector temporal Ask/Bid subyacente, refutando ejecuciones oportunistas a mercado.

### B. Invalidez de Truncamientos Dimensionales
*   **Función:** `test_gate6_mini_no_artificial_eom_truncation`
*   **Objetivo:** Asegura que una finalización intradiaria clasificada bajo el tipaje `ARTIFICIAL_TRUNCATION` impida la inclusión de la operación en las métricas consolidadas.

### C. Bloqueo Institucional Fail-Close
*   **Función:** `test_gate6_mini_news_missing_blocks_run`
*   **Objetivo:** Confirma la interrupción incondicional de la simulación ante el extravío de dependencias del calendario de noticias.

### D. Eliminación de Límites Silenciosos de Entrada
*   **Función:** `test_gate6_mini_v2b_stop_entry_window_not_silently_truncated`
*   **Objetivo:** Prueba que la caducidad de una orden Stop pendiente se determina mediante una fecha y hora de expiración real (`entry_deadline`), en lugar de cortes estáticos arbitrarios.

### E. Completitud Temporal del Truncamiento
*   **Función:** `test_gate6_mini_artificial_truncation_uses_window_completeness`
*   **Objetivo:** Vincula causalmente el estado de truncamiento a la falta física de datos observados al término de la sesión de trading esperada.

### F. Integridad de la Atribución N
*   **Función:** `test_n_attribution_completeness`
*   **Objetivo:** Verifica la consistencia matemática del embudo dimensional de rechazos.
