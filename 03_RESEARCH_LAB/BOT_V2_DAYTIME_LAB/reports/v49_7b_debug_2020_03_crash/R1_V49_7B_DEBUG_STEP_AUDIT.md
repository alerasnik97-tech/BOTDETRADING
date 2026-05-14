# R1 V49.7B DEBUG ?" STEP AUDIT

| Step | Duration (s) | Rows Out | Status |
| :--- | :--- | :--- | :--- |
| LOAD_CONFIGS | 0.01 | 5 | OK |
| LOAD_NEWS | 0.02 | 504 | OK |
| LOAD_TICKS | 0.41 | 5.7M | OK |
| BUILD_BARS | 0.92 | 6,370 | OK |
| BUILD_LEVELS | 0.11 | 27 | OK |
| DETECT_SIGNALS | 0.18 (avg) | ~300 | OK |
| EXECUTION | ~10.0 | 62 trades | OK |
| WRITE_RESULTS | 0.004 | 62 | OK |

**Veredicto**: Todos los pasos individuales son seguros. El riesgo es la acumulacin masiva en el loop de 800 configs.
