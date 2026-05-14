# PLANTEAMIENTO DEL PROBLEMA TÉCNICO (PROBLEM STATEMENT)
**Fase de Remediación:** V49.6 R1 Parameter Injection Remediation Gate  
**Módulos Afectados:** Orquestador de Expansión (`v49_expansion_run.py`) e Interfaz Adaptadora R1  
**Síntoma Físico:** Redundancia Muestral y Colisión Hash Transaccional  

---

## 1. Naturaleza del Fallo Forense
La auditoría independiente ejecutada en el hito V49.5 reveló una anomalía estructural severa en la fábrica de candidatos de la estrategia **R1**: configuraciones declarando intenciones operativas divergentes en su matriz de entrada (`entry_type` variando entre `NEXT_OPEN` y `MIDPOINT_STOP`) y modelos de detención teóricamente asimétricos (`sl_model` con literales de pips distintos) generaban invariablemente firmas transaccionales y curvas de equidad idénticas a nivel de bit.

## 2. Origen del Defecto en el Adaptador
La inspección en profundidad del orquestador activo `v49_expansion_run.py` expuso que la rotura reside puramente en la capa de traducción e invocación de señales hacia el motor central `UnifiedV7Engine`. Específicamente:
- **Omisión de Inyección de Entrada:** El bucle de procesamiento transaccional invoca sistemáticamente `engine.execute_signal(sig.direction.lower(), ts, t_window)` omitiendo transferir el atributo `cfg.entry_type` y forzando una ejecución incondicional a mercado (`entry_mode = "market"` implícito por defecto).
- **Sobrescritura Estática de Riesgo:** El pasaje de salidas sobrescribe incondicionalmente la distancia del stop loss con la constante procedimental `sl_mult = 1.5`, desestimando por completo el valor literal alojado en `cfg.sl_model`.
- **Desvinculación del Umbral BE:** La clase `R1Config` carece del atributo de Break-Even, impidiendo que el motor aplique cierres de protección dinámicos en las rebanadas intradiarias.

## 3. Consecuencias en el Espacio Dimensional
1. **Falsificación de Densidad:** El espacio de búsqueda sufre de un sesgo de aparente completitud, donde múltiples filas del grid son en realidad clones o espejos redundantes de una única primitiva subyacente.
2. **Pérdida de Variabilidad Intradiaria:** Al no honrarse las órdenes condicionales de tipo `Stop` ni los rechazos de límite, se invalida el escrutinio de absorción en la apertura neoyorquina, forzando al backtest a asumir entradas subóptimas y sesgadas.

## 4. Mandato Contractual de Remediación
Se impone refactorizar de forma rigurosa la lógica del orquestador en `v49_expansion_run.py` para asegurar que cada permutación paramétrica del grid instancie unívocamente las rutas causales correspondientes en `UnifiedV7Engine`, restaurando la fidelidad de simulación sin contaminar el núcleo inmutable de ejecución.
