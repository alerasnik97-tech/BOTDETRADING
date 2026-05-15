# P0/P1 No-Synthetic Policy (Anti-Placeholder)
Fecha: 2026-05-14

## Declaración de Principios
El laboratorio cuantitativo opera exclusivamente sobre evidencia física. Queda terminantemente prohibido el uso de datos sintéticos, generadores aleatorios o placeholders para la toma de decisiones estratégicas, validación de edge o reportes de cumplimiento.

## Reglas Obligatorias
1. **Origen de Datos**: Todo trade reportado debe originarse en parquets certificados del `05_MARKET_DATA_VAULT`.
2. **Prohibición de np.random**: No se permite el uso de `numpy.random` o librerías similares en scripts de resultados o runners oficiales.
3. **Prueba de Ejecución**: Cada trade debe contar con un `execution_id` trazable al log del motor (`CausalLog`).
4. **Recálculo de Métricas**: Ninguna métrica (PF, WR, DD) se acepta por narrativa; debe ser recalculable a partir de la lista de trades.
5. **Auditoría de Fechas**: Se requiere un split audit (Train/Val/Test) para asegurar que no hay solapamientos accidentales.
6. **No Test Leakage**: Queda prohibido cualquier acceso al set 2025-2026 fuera de la puerta final de aceptación.
7. **Verificación de Núcleo**: Todo resultado debe ser validado con el motor `UnifiedV7Engine` en su estado de lockdown.

## Incumplimiento
Cualquier violación a esta política invalidará de inmediato la estrategia bajo estudio, requiriendo un reinicio completo del proceso de investigación (Reset Gate).
