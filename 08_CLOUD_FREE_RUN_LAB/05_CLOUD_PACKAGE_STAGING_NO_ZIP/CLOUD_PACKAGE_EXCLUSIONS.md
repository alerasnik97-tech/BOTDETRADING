# CLOUD_PACKAGE_EXCLUSIONS

No incluir NUNCA en un paquete cloud:
- Carpeta `05_MARKET_DATA_VAULT` completa.
- Carpeta `07_BACKUPS`.
- Historial de `git` completo.
- Credenciales de brokers (incluso si están en archivos `.py`).
- Tokens de Telegram.
- Passwords de bases de datos.
- Archivos de resultados de otras estrategias.
- Cualquier dato que no sea estrictamente necesario para la corrida actual.
