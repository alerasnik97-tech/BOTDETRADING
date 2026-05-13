# POLÍTICA DE PREVENCIÓN DE ARTEFACTOS SINTÉTICOS (ANTI-PLACEHOLDER) — R1 V46

## 1. Mandatos de Autenticidad Física
Se prohíbe terminantemente la generación de reportes basados en datos sintéticos o esquemáticos que no correspondan bit-a-bit con la ejecución del motor.

### Reglas de Obligado Cumplimiento:
- **Paridad de Conteo**: Ningún CSV puede contener menos filas que lo declarado en la narrativa (Ej: Ranking 1200 configs = 1201 filas).
- **Consistencia Transaccional**: Ningún summary puede afirmar un número N mayor que las filas reales presentes en `R1_V46_TRADES.csv`.
- **Verificación Independiente**: Todo reporte de métricas debe provenir de un recálculo directo desde el archivo de trades, no de una repetición narrativa.
- **Trazabilidad de Ejecución**: Se requiere la inclusión de `R1_V46_RUN_LOG.txt` con marcas de tiempo reales de la corrida.
- **Veto de Muestra**: Si un archivo es una muestra (`SAMPLE`), debe estar claramente identificado y NO puede usarse como sustento para una decisión de paso a producción.

## 2. Consecuencias de la Discrepancia
Cualquier desajuste detectado entre los archivos físicos y los reportes resultará en el estado **R1_V46_BLOCKED_PLACEHOLDER_ARTIFACTS**.
