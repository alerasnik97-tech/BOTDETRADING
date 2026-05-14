# V50B RERUN REAL REQUIREMENTS

Para que una re-ejecucin de V50B sea aceptada, debe cumplir con:

1. **Detectores Operativos**: Implementacin real de las lGicas F01, F06, F08, F12 en `src/v50b_research_families/`.
2. **Ejecucin del Runner**: Uso de `v50b_family_preflight_runner.py` llamando al motor por cada mes y familia.
3. **Muestra Real**: Ejecucin sobre los 10 meses representativos definidos en el plan original.
4. **Logs del Motor**: Generacin de logs de ejecucin que muestren la lectura de parquets/ticks.
5. **Auditora de Recalculo**: Verificacin fsca de que los trades en el ranking coinciden con los trades individuales generados por el motor.
6. **No TEST Leakage**: Confirmacin de que no hay trades en 2025/2026.
7. **Engine Verify**: Estado `ENGINE_CORE_OK` antes y despuǸs de la corrida.

**Advertencia**: Cualquier indicio de datos sintǸticos en el rerun resultarǭ en la invalidacin inmediata del agente.
