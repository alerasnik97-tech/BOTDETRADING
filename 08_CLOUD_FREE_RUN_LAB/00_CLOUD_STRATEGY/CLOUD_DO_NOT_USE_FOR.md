# CLOUD_DO_NOT_USE_FOR

Este documento enfatiza las prohibiciones críticas para el uso de la nube gratuita:

1. **Datos Crudos Sensibles**: No subir archivos `.parquet` o `.csv` que contengan datos propietarios pesados o de difícil obtención.
2. **Secretos de Estado**: Nada de `.env`, `secrets.json` o variables de entorno con tokens de Telegram, API keys de brokers o passwords.
3. **Producción**: Jamás intentar conectar una cuenta real (ni siquiera demo con fondos significativos) a una instancia gratuita.
4. **Almacenamiento Permanente**: No usar la nube como backup. Los archivos en instancias gratuitas son volátiles.
5. **Abuso de Recursos**: No usar scripts que consuman el 100% de CPU de forma constante si el proveedor lo prohíbe (prevención de baneo).
6. **Decisiones Finales**: Los resultados de la nube son solo indicativos hasta que se repliquen localmente.
