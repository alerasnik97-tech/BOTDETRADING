# POLÍTICA DE PREVENCIÓN DE ARTEFACTOS SINTÉTICOS (ANTI-PLACEHOLDER) — R1

## 1. Mandato de Autenticidad Física
Se establece como violación grave de gobernanza la creación de reportes basados en datos alucinados o placeholders sintéticos.
- **Regla del Conteo Real**: Todo CSV debe contener el número exacto de filas declarado en la narrativa.
- **Regla de la Transaccionalidad**: El archivo de trades debe ser el reflejo bit-a-bit de la ejecución del motor, sin recortes manuales.
- **Regla de la Trazabilidad**: Se requiere la inclusión de logs de ejecución y archivos de configuración procesados para cada fase.

## 2. Protocolo de Verificación Cruzada
Ningún resultado se considerará válido si no supera la reconciliación entre:
1. El archivo de trades completo.
2. El ranking de configuraciones.
3. El reporte de métricas institucionales.

## 3. Sanción por Incumplimiento
Cualquier discrepancia mayor al 0.1% entre la narrativa y la evidencia física resultará en la invalidación inmediata de la fase (Estado RED).
