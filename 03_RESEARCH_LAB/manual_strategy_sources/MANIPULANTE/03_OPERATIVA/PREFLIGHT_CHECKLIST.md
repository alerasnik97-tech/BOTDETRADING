# PRE-FLIGHT CHECKLIST: MANIPULANTE LIVE OPERATIONS

## ANTES DE CADA SESIÓN (06:45 NY)
- [ ] **Terminal MT5:** Conectado y respondiendo (latency < 100ms preferible).
- [ ] **Cuenta FTMO:** Verificar que es la cuenta Demo correcta y el Auto-Trading está HABILITADO.
- [ ] **Spread EURUSD:** < 1 pip (ideal < 0.5 pips).
- [ ] **Balance:** Confirmar balance disponible > $9,000 para permitir riesgo 0.50% (lote mínimo).
- [ ] **Clean State:** Sin trades abiertos huérfanos del día anterior.
- [ ] **News Cache:** Archivos `ftmo_news_today.json` y `ftmo_news_week.json` están actualizados (correr script de news downloader si es necesario).
- [ ] **Logs Limpios:** Runner logs de las últimas 24h sin errores críticos (`runner.lock` stale removido si existiera).
- [ ] **Confirmation File:** `I_CONFIRM_FTMO_TRIAL_AUTO.txt` está presente en la carpeta correcta.

## DURANTE SESIÓN (07:00 - 16:30 NY)
- [ ] **Monitoreo Periódico:** Chequear `quick_status.txt` cada 30-60 minutos.
- [ ] **Conexión VPS/PC:** Asegurar que la PC no entre en suspensión ni pierda red.
- [ ] **Intervención Manual:** Sólo si ocurre un crash catastrófico de MT5. Si el bot marca "PELIGRO - NO APAGAR PC", respetar.

## AL CERRAR SESIÓN (19:50 NY)
- [ ] **Flat Verification:** Verificar que todos los trades cerraron efectivamente después del Forced Close de 19:45 NY.
- [ ] **Reconciliación:** Comparar PnL del archivo CSV de decisiones vs historial de MT5.
- [ ] **Backup:** Guardar un respaldo del archivo de decisiones de la sesión si fue una sesión con trades.
- [ ] **Cierre de Terminal:** Apagar terminal de MT5 de manera segura si la PC se apagará, para evitar "trades colgados" al día siguiente.
