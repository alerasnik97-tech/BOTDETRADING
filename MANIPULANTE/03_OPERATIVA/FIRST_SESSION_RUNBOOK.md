# RUNBOOK OPERATIVO: PRIMERA SESIÓN LIVE EN DEMO

## 1. PRE-FLIGHT CHECKLIST (06:45 NY)
- [ ] **MT5:** Conectado a la cuenta Demo FTMO, Auto-Trading HABILITADO.
- [ ] **Variables de Entorno:** Verificar que `TELEGRAM_BOT_TOKEN` y `TELEGRAM_CHAT_ID` estén correctamente configuradas a nivel OS.
- [ ] **News Cache:** Archivos json de noticias presentes en la carpeta de compliance.
- [ ] **Spread/Latency:** Verificar visualmente en MT5 que el spread en EURUSD sea < 1 pip y conexión < 100ms.
- [ ] **Clean State:** Ningún trade residual de días anteriores. No hay archivos `lock` bloqueando el runner.

## 2. INICIO DEL RUNNER (06:55 NY)
1. Abrir terminal en la raíz del proyecto.
2. Ejecutar el bat de inicialización o el script de python directamente:
   `python BOT_V2_DAYTIME_LAB\src\phase37_ftmo_trial_bot_runner.py`
3. Confirmar que el output indique: `STATE: ACTIVE` o `WAITING_FOR_WINDOW` si es antes de las 07:00 NY.

## 3. MONITOREO IN-SESSION (07:00 - 16:30 NY)
- **Frecuencia:** Cada 60 minutos.
- **Acción:** Abrir el archivo `quick_status.txt` (en `MANIPULANTE\10_LOGS_PAPER\ftmo_trial_bot\`).
- **Verificar:** `RUNNER=ACTIVO`, `MT5_CONNECTED=YES`.
- NO interactuar con MetaTrader a menos que haya una emergencia extrema (ej: bot no cerró en BE/TP).

## 4. PROTOCOLOS DE EMERGENCIA
- **MT5 Desconecta:** El bot quedará en loop infinito esperando conexión, suspendiendo el trading. Restaurar la conexión. Si un trade quedó abierto, monitorearlo en el teléfono.
- **Bot Crashea:** Revisar logs en terminal. Si hay posición abierta, el StopLoss nativo de MT5 es la salvaguarda. Ajustar TP/BE manualmente según protocolo si es necesario.
- **Telegram no responde:** Ignorar, el sistema core no depende de los mensajes de alerta (Fail-Open para Telegram, Fail-Closed para MT5).
- **Detener bot:** Borrar o detener el proceso python. Para freno blando de emergencia: Crear el archivo `STOP_BOT.txt` en la ruta especificada.

## 5. POST-SESIÓN (19:50 NY)
- **Reconciliación:** Comparar el CSV de decisiones (`reports\phase63c\PHASE63C_RESTORED_TRADES.csv`) o logs locales vs historial de MT5.
- **Flat:** Verificar posiciones=0.
- **Limpieza:** Apagar terminal y detener el loop para evitar arrastres nocturnos.

## 6. DEFINICIÓN DE ÉXITO (MES 1)
Para que el primer mes en Live Demo se considere EXITOSO, debe cumplir estrictamente con:
1. **0 violaciones de Risk Management** (Nadie operó >0.5% riesgo).
2. **0 trades fuera de la ventana** (Nada fuera de 07:00-16:30 NY).
3. **0 trades sin SL** (Toda orden fue con SL hardcodeado).
4. **100% resoluciones válidas** (Todo trade cerró por TP, BE, SL o el FORCED CLOSE de las 19:45 NY).
5. **PF Mensual > 0.8** (Permitiendo varianza normal del mes, pero demostrando que no hay un colapso total de expectativa).

*Cualquier fallo en los puntos 1 a 4 obliga a suspender operativas y realizar Phase64 (Post-Mortem Live).*
