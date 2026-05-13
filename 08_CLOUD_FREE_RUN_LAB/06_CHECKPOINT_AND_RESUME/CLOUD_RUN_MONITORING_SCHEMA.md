# CLOUD_RUN_MONITORING_SCHEMA

El monitoreo de las corridas gratuitas se basará en:
1. **Logs Locales (en la nube)**: Revisión periódica manual de archivos `run.log`.
2. **Webhooks de Estado (Opcional)**: Enviar un mensaje simple (sin datos sensibles) a un bot de Telegram cuando:
   - Inicie la corrida.
   - Ocurra un error crítico.
   - Finalice exitosamente.
   - Se guarde un checkpoint.
3. **Métricas de Salud**: Monitorear carga de CPU y uso de disco mediante scripts simples.
