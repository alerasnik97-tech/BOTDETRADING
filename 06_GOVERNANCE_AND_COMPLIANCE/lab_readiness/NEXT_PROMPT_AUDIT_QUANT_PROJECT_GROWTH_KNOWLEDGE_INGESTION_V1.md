# NEXT PROMPT - READ-ONLY AUDIT QUANT PROJECT GROWTH KNOWLEDGE INGESTION V1

Actua como auditor read-only, knowledge management officer, data governance officer y security/file-protection officer del proyecto Trading BOT.

Scope autorizado:

- Repository root: C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo
- Source folder: C:\Users\alera\Desktop\INVESTIGACIÓN QUANT
- Excluded folder: C:\Users\alera\Desktop\INVESTIGACIÓN QUANT\ESTRATEGIAS POSIBLES
- Destination folder: C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\03_RESEARCH_LAB\knowledge_intake\external_quant_project_growth_20260518\INVESTIGACION_QUANT_GENERAL
- Manifest: 03_RESEARCH_LAB/knowledge_intake/external_quant_project_growth_20260518/INVESTIGACION_QUANT_GENERAL_MANIFEST.csv
- Index: 03_RESEARCH_LAB/knowledge_intake/external_quant_project_growth_20260518/INVESTIGACION_QUANT_GENERAL_INDEX.md
- Report: 03_RESEARCH_LAB/knowledge_intake/external_quant_project_growth_20260518/INVESTIGACION_QUANT_GENERAL_INGESTION_REPORT.md

Modo obligatorio: SOLO LECTURA. No modificar, mover, borrar, copiar, ejecutar, renombrar, normalizar, OCR, resumir contenido, backtest, train, validation, holdout, optimization o sweep.

Auditar y reportar:

1. Origen no borrado y no modificado.
2. Destino existe y copia completa.
3. ESTRATEGIAS POSIBLES quedo excluida y documentada como strategy-research-intake handled separately.
4. Manifest correcto: paths, tamanos, fechas, hashes source/copy, status y category_guess.
5. Todos los hashes coinciden; si no, BLOCKED_HASH_MISMATCH.
6. PDFs/binarios/documentos fuente no commiteados.
7. No codigo modificado.
8. No tests modificados.
9. No datos de mercado modificados.
10. No backtest/train/validation/holdout ejecutado.
11. No uso de 2025/2026.
12. No optimization/sweep.
13. No secrets expuestos ni nombres de archivo sospechosos en artefactos versionados.
14. No outputs prohibidos ni ZIPs creados.
15. .gitignore local correcto.
16. Git staging/commit contiene solo markdown/csv/gitignore autorizados.

Formato de salida requerido:

- STATUS: READY / BLOCKED / INCONCLUSIVE
- SOURCE_INTACT: YES/NO
- EXCLUSION_APPLIED: YES/NO
- COPY_COMPLETE: YES/NO
- HASH_MISMATCHES: count
- BINARY_DOCS_COMMITTED: YES/NO
- CODE_OR_TESTS_MODIFIED: YES/NO
- MARKET_DATA_MODIFIED: YES/NO
- EXECUTION_DETECTED: YES/NO
- FORBIDDEN_OUTPUTS_DETECTED: YES/NO
- DECISION:
- EVIDENCE_PATHS: