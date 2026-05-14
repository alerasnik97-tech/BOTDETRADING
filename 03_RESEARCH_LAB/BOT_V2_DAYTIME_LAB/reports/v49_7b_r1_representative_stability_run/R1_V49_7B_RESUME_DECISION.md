# R1 V49.7B ?" RESUME DECISION

**Estado Final**: V49_7B_RESTART_CLEAN_REQUIRED

## Justificacin
1. **Diferencia de Alcance**: V49.7 era una mini-corrida (100 configs, 2 meses). El requerimiento actual es Representative Stability Run (800+ configs, 2020-2024).
2. **Falta de Checkpoints**: El runner `v49_7_overnight_runner.py` no implementa guardado incremental. Un resume implicaría reprocesar todo o riesgo de datos incompletos.
3. **Integridad de Evidencia**: Para cumplir con el protocolo de Calidad X10, una corrida limpia con el grid reparado y el alcance completo es la única opción válida.

## Prximos Pasos
1. Archivar parciales de V49.7.
2. Desplegar `v49_7b_full_scope_runner.py` con soporte para resume (opcional) y alcance completo.
3. Ejecutar Engine Verify.
4. Lanzar corrida V49.7B.
