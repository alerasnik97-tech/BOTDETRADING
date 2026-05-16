# MANIPULANTE - DOCUMENTO MAESTRO DEL BOT OFICIAL

Este es el documento de identidad oficial del bot **MANIPULANTE**, el primer bot de trading consolidado y validado del laboratorio.

## 1. Identidad y Autoridad
- **Nombre Oficial**: MANIPULANTE.
- **Tipo**: Bot EURUSD diurno de alta precision.
- **Estado**: **BOT OFICIAL #1** (Consolidado).
- **Autoridad**: MANIPULANTE (Phase 25).
- **Origen Historico**: Derivado de la validacion profunda Phase 25/27.

## 2. Parametros Oficiales (Bloqueados)
| Parametro | Valor |
| :--- | :--- |
| **Simbolo** | EURUSD |
| **Contexto** | H1 Fractal Sweep |
| **Entrada** | First M3 CHOCH |
| **Take Profit (TP)** | 1.4R |
| **Break Even (BE)** | 0.4R (Trigger) |
| **Body Filter (BF)** | 70% |
| **Ventana Operativa** | 07:00 – 16:30 NY |
| **Frecuencia Maxima** | 1 trade por dia |
| **Riesgo Permitido** | 0.50% (FTMO Trial) |
| **Riesgo Prohibido** | 1.00% o superior |

## 3. Gestion Operativa y Seguridad
- **News Fortress**: El bot se bloquea automaticamente antes, durante y despues de noticias de alto impacto (FAIL-CLOSED).
- **Data Quality Mask**: Verifica la integridad de la data de MT5 antes de permitir cualquier operacion.
- **OrderSend Gateway**: Las ordenes pasan por un filtro de validacion de cuenta (FTMO Demo Only).
- **Control de Procesos**: Ciclo seguro mediante **START / STATUS / STOP**.
- **Cierre Diario**: Forced Close automatico a las 19:45 NY.
- **Hard Close Semanal**: Cierre obligatorio Viernes 16:55 NY (Sin excepciones).

## 4. Rutina Diaria del Operador
1.  **Conexion**: Abrir MT5 y confirmar cuenta **FTMO Demo**.
2.  **Activacion**: Presionar "Trading algoritmico" en MT5 (Icono en VERDE).
3.  **Encendido**: Ejecutar `START_MANIPULANTE.bat` y dejar la ventana minimizada.
4.  **Monitoreo**: Consultar `STATUS_MANIPULANTE.bat` periodicamente.
5.  **Cierre**: Ejecutar `STOP_MANIPULANTE.bat` al finalizar el dia o tras el cierre de posiciones.
6.  **Seguridad**: No apagar la PC hasta que STATUS confirme `SEGURO APAGAR PC: SI`.

## 5. Estado Actual y Proxima Meta
**MANIPULANTE** se encuentra en fase de **Forward Demo Disciplinado**. 
- No se permiten modificaciones de parametros por "ansiedad" o rachas negativas.
- El bot ha sido auditado por costos reales (comisiones y spread) y es rentable.
- Meta: Completar la muestra estadistica en Demo antes de la compra de evaluacion real.

---
*Documento consolidado en la Phase 39.*
