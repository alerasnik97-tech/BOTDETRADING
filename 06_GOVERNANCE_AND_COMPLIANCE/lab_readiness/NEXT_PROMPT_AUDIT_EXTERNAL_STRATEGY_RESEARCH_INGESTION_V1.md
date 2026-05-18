# NEXT PROMPT - READ-ONLY AUDIT EXTERNAL STRATEGY RESEARCH INGESTION V1

Actua como auditor read-only, data governance officer y security/file-protection officer del proyecto Trading BOT.

Scope autorizado:

- Repository root: C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo
- Source folder: C:\Users\alera\Desktop\INVESTIGACIÓN QUANT\ESTRATEGIAS POSIBLES
- Destination folder: C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\strategy_research_intake\external_research_20260518\ESTRATEGIAS_POSIBLES
- Manifest: 03_RESEARCH_LAB/strategy_research_intake/external_research_20260518/ESTRATEGIAS_POSIBLES_MANIFEST.csv
- Index: 03_RESEARCH_LAB/strategy_research_intake/external_research_20260518/ESTRATEGIAS_POSIBLES_INDEX.md
- Report: 03_RESEARCH_LAB/strategy_research_intake/external_research_20260518/ESTRATEGIAS_POSIBLES_INGESTION_REPORT.md

Modo obligatorio: SOLO LECTURA. No modificar, mover, borrar, copiar, ejecutar, renombrar, normalizar, OCR, backtest, train, validation, holdout, optimization o sweep.

Auditar y reportar:

1. Origen no borrado y no modificado.
2. Destino existe y copia completa.
3. Manifest correcto: paths, tamaños, fechas, hashes source/copy y status.
4. Todos los hashes coinciden; si no, BLOCKED_HASH_MISMATCH.
5. PDFs/binarios/documentos fuente no commiteados salvo politica explicita posterior.
6. No codigo modificado.
7. No tests modificados.
8. No datos de mercado modificados.
9. No backtest/train/validation/holdout ejecutado.
10. No uso de 2025/2026.
11. No optimization/sweep.
12. No secrets expuestos ni nombres de archivo sospechosos en artefactos versionados.
13. No outputs prohibidos ni ZIPs creados.
14. Git staging/commit contiene solo markdown/csv/gitignore autorizados.

Formato de salida requerido:

- STATUS: READY / BLOCKED / INCONCLUSIVE
- SOURCE_INTACT: YES/NO
- COPY_COMPLETE: YES/NO
- HASH_MISMATCHES: count
- BINARY_DOCS_COMMITTED: YES/NO
- CODE_OR_TESTS_MODIFIED: YES/NO
- MARKET_DATA_MODIFIED: YES/NO
- EXECUTION_DETECTED: YES/NO
- FORBIDDEN_OUTPUTS_DETECTED: YES/NO
- DECISION:
- EVIDENCE_PATHS: