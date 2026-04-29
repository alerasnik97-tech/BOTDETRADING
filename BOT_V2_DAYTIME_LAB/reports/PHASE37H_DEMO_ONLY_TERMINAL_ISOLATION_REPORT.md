# PHASE37H DEMO-ONLY TERMINAL ISOLATION REPORT

## 1. Objetivo
Aislar el terminal FTMO Demo y restaurar la automatización de noticias tras la detección de una cuenta real de Exness en Phase 37G.

## 2. Terminal Discovery
- **Terminal Activo**: C:\Program Files\MetaTrader 5
- **Data Path**: C:\Users\alera\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075
- **Company**: FTMO Global Markets Ltd
- **Server**: FTMO-Demo
- **Account Mode**: DEMO (0)
- **Balance**: 10000.0 USD
- **Clasificación**: **FTMO_DEMO_ALLOWED**

## 3. Account Gate
- **FTMO Demo/Trial Confirmado**: SÍ.
- **Exness Detectado**: NO (en la sesión activa).
- **Real Detectado**: NO (en la sesión activa).
- **Allowlist Pass**: SÍ.

## 4. Safety Recovery
- **Cambios detectados**: Algo Trading estaba desactivado por seguridad tras el cambio de cuenta.
- **Restaurados**: SÍ. Se re-habilitó `Enabled=1` en `common.ini` y se re-inyectó el EA en `chart01.chr`.
- **Nota**: MT5 build 5833 auto-desactiva el botón de "Algo Trading" si detecta un cambio de broker/cuenta, lo cual bloquea la ejecución automática inicial.

## 5. MQL5 Safety
- **Componentes auditados**: SÍ.
- **Funciones trading**: NINGUNA.
- **Veredicto**: PASS.

## 6. News Status
- **Cache Hoy**: MISSING.
- **Cache Semana**: MISSING.
- **News Gate**: NO_TRADE.

## 7. Full Gate Rerun
- Account: PASS.
- News: **FAIL** (Cache missing).
- Data/Time/Symbol/Lot: PASS.
- Signal: SYNC_OK.
- STOP_BOT: **ACTIVE**.
- Confirmation: ABSENT.

## 8. Veredicto Final
**FTMO_TRIAL_REQUIRES_ONE_TIME_BOOTSTRAP**

## 9. Instrucción de Bootstrap Único (Manual)
Debido a las protecciones de seguridad de MT5 build 5833 tras el cambio de cuenta:
1. Abra el terminal MT5 (FTMO Demo).
2. Asegúrese de que el botón **"Algo Trading"** esté presionado (verde).
3. Arrastre `MANIPULANTE_CalendarBootstrapEA.ex5` (en Experts\MANIPULANTE) a un gráfico de EURUSD.
4. Una vez generado el archivo de caché (menos de 30 segundos), el sistema podrá operar 100% automático.

## 10. Seguridad
- No se detectaron riesgos en la cuenta real de Exness durante esta fase.
- El sistema permanece bloqueado (`STOP_BOT`) hasta que el News Gate pase.
