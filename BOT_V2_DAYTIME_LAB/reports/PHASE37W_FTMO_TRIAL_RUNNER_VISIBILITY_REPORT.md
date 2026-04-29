# PHASE 37W FTMO TRIAL RUNNER VISIBILITY & CONTROL PANEL REPORT

## 1. Objetivo
Implementar visibilidad total, control manual y protección contra duplicados para el auto-runner de MANIPULANTE en FTMO Trial.

## 2. Veredicto Final Exacto
**FTMO_TRIAL_RUNNER_VISIBLE_AND_HEALTHY**

## 3. MT5 Status
- **Abierto**: SÍ.
- **FTMO Demo**: SÍ (FTMO Global Markets Ltd).
- **Exness Detectado**: NO.
- **Real Detectado**: NO.

## 4. Runner Status
- **Activo**: SÍ.
- **PID**: 16404.
- **Protección**: `runner.lock` activo (impide múltiples instancias).
- **Loop Continuo**: SÍ (60 segundos).

## 5. Heartbeat
- **Creado**: SÍ (`heartbeat.json` / `heartbeat.txt`).
- **Última actualización**: 12:16:55 NY.
- **Estado**: **RUNNING / ALLOW**.

## 6. Última Decisión
- **Veredicto**: **ALLOW** (Signal Ready).
- **Motivo**: Todos los gates pasan y se detectó setup LONG a las 09:18 NY.

## 7. Próxima Ventana de Noticias
- **Evento**: **AIE Cambio en las Reservas de Crudo (USD)**.
- **Horario NY**: 13:30 NY.
- **Ventana de Guardia**: **13:00 NY - 14:00 NY**.

## 8. Panel de Control (Scripts .bat)
Se han creado los siguientes scripts en `MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\`:
- **START_FTMO_TRIAL_AUTO.bat**: Valida cuenta FTMO, verifica bloqueos e inicia el runner visible.
- **STOP_FTMO_TRIAL_AUTO.bat**: Crea el kill-switch `STOP_BOT.txt` para cierre seguro.
- **STATUS_FTMO_TRIAL_AUTO.bat**: Muestra PID, Heartbeat y últimas decisiones en 5 segundos.

## 9. Blockers / Warnings
- Ninguno. El sistema está operando de forma autónoma y transparente.

## 10. Confirmación de Seguridad
- **No Real**: SÍ.
- **No Exness**: SÍ.
- **No Secretos**: SÍ.
- **No Estrategia Modificada**: SÍ.

## 11. Siguiente Paso Único
**Uso de Scripts**: El usuario puede ahora controlar y auditar el bot usando los archivos `.bat` creados, sin necesidad de comandos manuales complejos.
