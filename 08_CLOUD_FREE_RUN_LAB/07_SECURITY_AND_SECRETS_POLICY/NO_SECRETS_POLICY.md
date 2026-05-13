# NO_SECRETS_POLICY

1. **PROHIBICIÓN ABSOLUTA**: No se permite la subida de ningún archivo `.env`, `.json` o cualquier formato que contenga:
   - API Keys de brokers (OANDA, MetaTrader, Binance, etc.).
   - Tokens de Telegram.
   - Contraseñas de bases de datos.
   - SSH Keys privadas del proyecto local.
2. **OFUSCACIÓN**: Si el código requiere una variable de entorno para funcionar, se debe pasar de forma manual en la sesión de la nube y nunca quedar persistida en el disco de la instancia.
3. **VERIFICACIÓN**: Antes de cada subida de paquete cloud, se debe realizar un escaneo manual o automático para detectar secretos.
