# V50B REAL QA ?" OBJECTIVE

**Meta**: Corregir la interpretación del Precheck Real y asegurar que las 4 familias candidatas tengan un pipeline de ejecución auditado, no solo de señales.

## Objetivos Especficos
1. **Auditora de Cobertura**: Verificar por qué F01, F06 y F08 no produjeron trades en el micro-run anterior.
2. **Rejection Probe**: Ejecutar un micro-run dedicado para capturar los `rejection_reason` del motor para las familias silenciosas.
3. **Validacin de Calendario**: Analizar el impacto del uso de `DummyNews` y planificar la conexión con noticias reales.
4. **Correccin de Estado**: Emitir una decisión global ajustada a la evidencia física real (Partial Pass).

**Veredicto Esperado**: Certificación de que el motor "escucha" y "decide" sobre todas las familias, con motivos de rechazo explícitos.
