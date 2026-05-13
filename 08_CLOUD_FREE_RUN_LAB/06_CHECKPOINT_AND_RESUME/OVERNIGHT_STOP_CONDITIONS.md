# OVERNIGHT_STOP_CONDITIONS

Detener inmediatamente la corrida cloud si:

- **RAM > 85%**: Riesgo de OOM (Out Of Memory) que cuelgue la instancia.
- **Disco libre < 10GB**: Riesgo de no poder guardar outputs o checkpoints.
- **Error repetido > 5**: Si el runner falla 5 veces seguidas en la misma tarea, abortar.
- **News calendar missing**: Si se requiere filtro de noticias y el archivo no está disponible.
- **Data gap**: Detección de huecos de datos no esperados en el dataset cloud.
- **EOM artificial en métricas**: Resultados que parecen físicamente imposibles.
- **Más de 3 trades/día**: Superación del límite de frecuencia diaria para esta fase.
- **TEST leakage detectado**: Cualquier indicio de que se está viendo el futuro.
- **PF_val muy bajo**: Early stop si el Profit Factor cae por debajo de un umbral en una muestra representativa.
- **FTMO blown sistemático**: Si la estrategia quema la cuenta simulada repetidamente.
- **Archivo output corrupto**: Fallo en la integridad de los resultados parciales.
- **Runner hash cambia**: Si por alguna razón el código de ejecución se modifica durante la corrida.
