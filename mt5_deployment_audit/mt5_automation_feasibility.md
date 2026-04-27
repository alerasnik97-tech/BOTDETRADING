# MT5 Automation Feasibility Audit

**Veredicto:** `ONLY_SAFE_AS_MANUAL_OR_SEMI_AUTOMATIC`

## 1. Análisis de la Lógica Real
Tras auditar `baseline_truth_model.py` y los archivos de configuración del candidato shadow, se determinan los siguientes puntos de fricción para una automatización 100% desatendida en MT5:

- **Dependencia de Niveles de Sesión:** La estrategia utiliza niveles de Asia (definidos a las 02:00 NY) y London (definidos a las 08:00 NY). Para operar correctamente, el bot debe tener los niveles calculados con precisión quirúrgica, incluyendo el "Sunday Fix" recientemente implementado.
- **Ventana de Escaneo Indefinida:** La lógica institucional no limita las horas de búsqueda de sweeps. Un sweep de un nivel de Asia puede ocurrir durante la sesión de Londres (03:00-08:00 NY) o durante la sesión de NY (08:00-17:00 NY). Un bot con ventana horaria rígida perdería oportunidades valiosas o contexto crítico.
- **Gestión de Salida (Timeout 4h):** El uso de un timeout temporal de 4 horas implica que el sistema debe permanecer activo para gestionar la salida por tiempo. Si se define una ventana de apagado a las 12:00 NY pero se entra a las 11:00 NY, el bot quedaría "huérfano" si se apaga antes de las 15:00 NY.

## 2. Bloqueo Institucional
El `micro_pilot_gate` se encuentra actualmente en estado `NOT_READY_FOR_MICRO_PILOT` debido a que la muestra de evidencia shadow es insuficiente (`N < 10`). La automatización completa en MT5 introduce riesgos de infraestructura (latencia, ejecución, conectividad) que no deben asumirse hasta que la lógica base haya demostrado robustez en la fase shadow/manual.

## 3. Conclusión de Auditoría
La estrategia **no admite una traducción segura a un bot 100% automático con ventana horaria fija** bajo el perfil de riesgo actual del proyecto. La naturaleza de los sweeps y la gestión por tiempo requieren una supervisión o una lógica de EA mucho más compleja de la que permite un despliegue simple por horario.

---
**Generado por Auditoría Técnica Antigravity - 2026-04-24**
