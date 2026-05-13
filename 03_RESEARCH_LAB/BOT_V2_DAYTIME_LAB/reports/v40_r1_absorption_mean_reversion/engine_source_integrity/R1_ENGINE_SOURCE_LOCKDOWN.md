# PROTOCOLO DE BLOQUEO INSTITUCIONAL (LOCKDOWN): R1 ENGINE SOURCE INTEGRITY

## Estado del Bloqueo
**ACTIVO — STRICT LOCKDOWN**

## Motivo
Violación de la integridad del código fuente del motor de simulación cuantitativa (`UnifiedV7Engine` y `v6_utils`). El uso de código reconstruido desde contexto impide la certificación de causalidad y correcta aplicación de costos FTMO.

## Restricciones Inmediatas (Prohibiciones)
- **Ejecución**: Prohibido iniciar o reanudar el barrido walk-forward de 76 meses de la estrategia R1.
- **Promoción**: Prohibido empaquetar o exportar la estrategia a entornos de nube (Kaggle, Oracle, Colab) o staging.
- **Evidencia**: Prohibido regenerar el archivo `000_PARA_CHATGPT.zip` oficial con resultados derivados de este motor divergente.
- **Git**: Prohibido realizar `git commit` o `git push` de los archivos del motor en el estado actual.

## Acciones de Desbloqueo Permitidas
1. Ejecutar auditoría forense de diffs y documentar hallazgos.
2. Aplicar el plan de restauración canónica extrayendo las fuentes inmutables desde Git (`agent/research-manipulante4-sweep-quality`).
3. Ejecutar las suites de tests automatizados (targeted y full suite) para demostrar estabilidad.
4. Purgar los resultados parciales contaminados y reiniciar un smoke test de preflight ultra-corto (1-2 meses).

## Condición de Salida (Exit Gate)
El bloqueo solo se levantará tras la firma exitosa del reporte de readiness final (`R1_ENGINE_FINAL_READINESS.md`) certificando el estado `R1_ENGINE_RESTORED_READY_FOR_CLEAN_PREFLIGHT`.
