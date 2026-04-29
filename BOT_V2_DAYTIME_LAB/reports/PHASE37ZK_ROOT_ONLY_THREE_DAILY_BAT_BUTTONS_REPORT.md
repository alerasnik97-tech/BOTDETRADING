# PHASE37ZK ROOT ONLY THREE DAILY BAT BUTTONS REPORT

## 1. Lo mas importante
Se ha simplificado la estructura de la carpeta **MANIPULANTE**, dejando visibles en la raiz unicamente los 3 botones principales para la operacion diaria: **START**, **STATUS** y **STOP**. El boton STOP ha sido dotado de inteligencia para prevenir el apagado si hay operaciones abiertas y asegurar una detencion limpia del runner.

## 2. Veredicto final exacto
**ROOT_THREE_BAT_BUTTONS_READY**

## 3. BATs finales en raiz
- **START_MANIPULANTE.bat**: Inicia el bot (con validacion de duplicados y bloqueos).
- **STATUS_MANIPULANTE.bat**: Muestra el panel de estado limpio (Phase 37ZJ).
- **STOP_MANIPULANTE.bat**: Detiene el bot de forma segura.

## 4. START
- Utiliza rutas absolutas.
- Detecta si el bot ya esta prendido y evita duplicados.
- Detecta si existe un bloqueo por `STOP_BOT.txt` y solicita confirmacion al usuario para reiniciarlo, borrando el bloqueo automaticamente tras la confirmacion.

## 5. STATUS
- Muestra el panel limpio actualizado cada 30 segundos.
- Traducido totalmente al español simple.
- No afecta la ejecucion del bot al cerrarse.

## 6. STOP
- **Inteligente**: Consulta el estado JSON del bot antes de proceder.
- **Seguro**: Si detecta `OPERACION_ABIERTA: SI`, bloquea la detencion automatica y emite una alerta de **PELIGRO**.
- **Limpio**: Si es seguro, crea la señal `STOP_BOT.txt`, espera 20 segundos y, si el runner no cerro solo, finaliza el proceso de forma selectiva sin tocar MT5.

## 7. Archivos movidos a archivo
- Se creo la carpeta `MANIPULANTE\99_ARCHIVO_BAT_ANTIGUOS\`.
- Se movio `STATUS_TECNICO_MANIPULANTE.bat` a esta carpeta para reducir el ruido en la raiz.

## 8. Tests
1. **START duplicado**: Validado (Avisa que ya esta prendido).
2. **STATUS**: Validado (Muestra info limpia).
3. **STOP sin posicion**: Validado (Crea señal, espera y confirma detencion).
4. **START con bloqueo**: Validado (Detecta STOP_BOT, pide tecla para borrarlo e inicia).
5. **Seguridad de Procesos**: Confirmado que STOP no mata a MT5 ni a procesos ajenos.

## 9. Seguridad
- **no real**: No se modificaron credenciales ni acceso a cuentas reales.
- **no Exness**: No se toco configuracion de brokers.
- **no estrategia modificada**: La logica de trading (Phase 25) permanece intacta.
- **no orden enviada**: Todas las pruebas se realizaron sobre la lectura de estado y gestion de procesos.

## 10. ZIP/Git
- ZIP canonico actualizado con la nueva estructura.
- Commit: `Phase37ZK root only three daily bat buttons`
- Push realizado a `main`.

## 11. Siguiente paso unico
Operar exclusivamente usando los 3 botones de la raiz de MANIPULANTE.
