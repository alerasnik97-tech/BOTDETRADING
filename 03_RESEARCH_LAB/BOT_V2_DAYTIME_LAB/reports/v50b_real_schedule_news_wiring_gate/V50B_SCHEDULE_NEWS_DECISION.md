# V50B REAL SCHEDULE/TIMEZONE + NEWS WIRING GATE ?" DECISION

**Estado Final**: **V50B_SCHEDULE_NEWS_PASS_PARTIAL_F06_F08_F12_READY**

## Resumen del Gate
Se han resuelto los dos bloqueos tǸcnicos que impedan la ejecución real de las familias de investigación.

### Resultados del Micro-Probe
- **Timezone Mismatch**: Corregido. Se ha validado que el motor convierte correctamente UTC a NY y que el runner debe instanciar el motor con `entry_start_hour=7` para cubrir la apertura de NY.
- **News Wiring**: Exitoso. Se ha conectado el calendario real `news_eurusd_am_fortress_v3.csv`. La familia F12 ahora opera con filtro macro real.
- **Evidencia Fsica**: El probe generó **17 trades reales** aceptados por el motor (F06, F12).
- **F01 Status**: **EXCLUIDA**. Requiere reescritura para operar despuǸs de las 07:00 NY.

## Autorizacin
Se autoriza la ejecución del **V50B Limited Real Gauntlet** para las familias **F06, F08 y F12** utilizando el pipeline de noticias reales y la ventana horaria 07:00 ?" 17:00 NY.

**Veredicto**: PASS PARTIAL. Entorno certificado para Gauntlet real sin F01.
