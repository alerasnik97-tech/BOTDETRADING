# PHASE37F MQL5 CALENDAR SERVICE EXPORTER REPORT

## 1. Objetivo
Resolver la extracción automática de noticias gratuitas desde MT5/MQL5 para el sistema MANIPULANTE en modo FTMO Trial.

## 2. Diagnóstico Phase37E
- **Fallo**: El Script Exporter no se compiló/ejecutó vía CLI.
- **Causa**: Problemas de rutas de inclusión y permisos de ejecución en el terminal build 5833.
- **Corrección**: Se migró a un **Service Exporter** y un **Bootstrap EA** con configuración `.ini` refinada.

## 3. MQL5 Calendar Service
- **Creado**: SÍ.
- **Ubicación**: `MQL5\Services\MANIPULANTE\MANIPULANTE_CalendarServiceExporter.mq5`
- **Funciones Trading**: NO encontradas (Audit PASS).
- **Compilado**: SÍ (OK).
- **.ex5 generado**: SÍ.

## 4. Service Autostart / Bootstrap
- **Mecanismo**: Se intentó lanzar el terminal con `/config` apuntando a un **Bootstrap EA**.
- **Resultado**: El terminal inicia y reconoce el config, pero **no carga el EA** ni genera la caché.
- **Bloqueo**: La versión actual de MT5 (build 5833) parece bloquear la ejecución automática de EAs/Scripts vía CLI si el usuario no ha interactuado previamente o si existen bloqueos de seguridad en el perfil.

## 5. News Cache & Gate
- **Hoy cargado**: NO.
- **Semana cargada**: NO.
- **Fuente**: MT5/MQL5 (Fallo en autorun).
- **Estado News Gate**: NO_TRADE.

## 6. Signal Revalidation
- **Estado**: MANIPULANTE_SIGNAL_SYNC_OK (TP 1.4, BE 0.4, BF 0.7).

## 7. Full Gate Rerun
- Account: PASS (FTMO_DEMO_TRIAL_CONFIRMED)
- Service: FAIL (Autorun blocked)
- News: NO_TRADE
- Data/Time/Symbol/Lot: ALLOW
- Signal: MANIPULANTE_SIGNAL_SYNC_OK
- Order Router: PASS
- STOP_BOT: ACTIVE
- Confirmation: ABSENT

## 8. Veredicto Final
**FTMO_TRIAL_BLOCKED_MQL5_SERVICE_AUTORUN**

El sistema es técnicamente capaz y seguro, pero la barrera de MT5 para el autostart de programas MQL5 impide la automatización al 100% sin intervención manual.

## 9. Siguiente Paso Único
**Intervención Manual Mandatoria**: El usuario debe arrastrar `MANIPULANTE_CalendarBootstrapEA.ex5` a un gráfico de EURUSD en MT5 **una vez** para generar la caché inicial. Luego, el bot runner podrá operar automáticamente.
