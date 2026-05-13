# REQUERIMIENTO OBLIGATORIO DE VERIFICACIÓN PREFLIGHT (RUN PREFLIGHT REQUIREMENT)

## 1. Norma General de Ejecución
Como medida de endurecimiento definitivo, la verificación criptográfica in situ del motor central pasa a ser una **condición sine qua non** (requisito previo estricto) antes de destinar capacidad de cómputo a simulaciones de larga duración.

## 2. Reglas de Validación por Entorno
- **Estrategia R1 (Full Run Local)**: Queda totalmente prohibido iniciar el ciclo walk-forward de 76 meses en la máquina local si el orquestador no valida con antelación que la llamada a `ENGINE_CORE_VERIFY.py` concluye con el estado de integridad `ENGINE_CORE_OK`.
- **Ejecuciones en la Nube (Kaggle / Oracle / Cloud Lab)**: Ningún empaquetado o contenedor efímero podrá dar curso a un backtesting masivo si no integra en su propia rutina de inicialización (bootstrapping script) la ejecución del script de verificación contra el manifiesto canónico.
- **Estrategias Futuras**: Cualquier nuevo motor de barrido de parámetros, detector de señales o micro-probe implementado por los agentes en el futuro hereda de forma obligatoria la imposición de incorporar este paso preflight en su capa de orquestación.

## 3. Comportamiento Exigido ante Fallo (Fail-Close Execution Guard)
Si el script de verificación detecta la más mínima divergencia (drift), archivos faltantes o intrusión de lógicas espurias (Intruder files), el orquestador está forzado a:
1. **Abortar incondicionalmente** la ejecución antes de invocar la capa de carga de datos de mercado.
2. **Bloquear la generación** de cualquier archivo `.csv`, `.json` o reporte de métricas parciales.
3. Emitir una traza de error visible alertando sobre el compromiso de la Fuente de Verdad.
