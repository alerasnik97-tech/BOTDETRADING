# MANIPULANTE - Sistema de Alertas Gratuitas (Telegram / Email)

## 1. Que son las alertas
Este sistema monitorea el estado del bot MANIPULANTE de forma pasiva (read-only) y envia notificaciones cuando ocurren eventos importantes.

## 2. Que pueden avisar
- **Criticos**: Bot apagado en sesion, MT5 desconectado, cuenta REAL detectada, errores de envio de orden.
- **Informativos**: Bot bloqueado por noticias, operacion abierta (Demo), seguro apagar PC.

## 3. Que NO hacen
- **NO** envian ordenes.
- **NO** tocan la estrategia.
- **NO** modifican el riesgo.
- **NO** reinician MT5.

## 4. Como configurar Telegram
1. Crear un bot con @BotFather en Telegram.
2. Obtener el `TOKEN`.
3. Obtener tu `CHAT_ID` (usando @userinfobot o similar).

## 5. Como configurar variables de entorno en Windows
Abrir CMD (como Administrador preferentemente) y ejecutar:
```cmd
setx TELEGRAM_BOT_TOKEN "TU_TOKEN_AQUI"
setx TELEGRAM_CHAT_ID "TU_CHAT_ID_AQUI"
```
**IMPORTANTE**: Reiniciar la terminal (o el PC) para que los cambios surtan efecto.

## 6. Como mandar test seguro
Ejecutar:
```bash
python BOT_V2_DAYTIME_LAB\src\phase45_telegram_sender.py --send-test
```

## 7. Como correr una vez
Usar el archivo BAT:
`MANIPULANTE\16_OBSERVABILITY\alerts\RUN_ALERTS_ONCE_MANIPULANTE.bat`

## 8. Como correr en loop
Usar el archivo BAT:
`MANIPULANTE\16_OBSERVABILITY\alerts\RUN_ALERTS_LOOP_MANIPULANTE.bat`

## 9. Como apagar alertas
Cerrar la terminal que corre el loop. O editar `alerts_config.local.json` y poner `telegram_enabled: false`.

## 10. Seguridad
- **NO** subir tokens a GitHub.
- **NO** compartir el archivo `alerts_config.local.json`.
- **Revocar** el token en @BotFather si crees que se filtro.
- El sistema es estrictamente **Read-Only**.
