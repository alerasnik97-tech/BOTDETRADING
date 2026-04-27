# MANUAL_VS_BOT_GAP_ANALYSIS.md

## 1. El Edge Manual Revelado
El análisis de 841 trades manuales (2020-2026) confirma un **Profit Factor de 1.88** y una **Esperanza de 0.36R**. Estos resultados son institucionales y superan drásticamente a cualquier bot evaluado anteriormente.

## 2. La Brecha Crítica: El Factor "Tiempo"
- **Usuario Manual:** El 100% de los trades ocurren entre las **08:00 y las 11:00 NY**.
- **Bot V1/V2 Research:** Intentaron operar desde las 07:00 hasta las 20:30 NY.
- **Conclusión:** El bot diluyó el edge al tomar señales en horarios de baja volatilidad (tarde de NY) que el usuario manual simplemente ignora.

## 3. La Brecha Técnica: Confirmación LTF
- El usuario opera principalmente en **3M**. Los bots de investigación previa usaron mayoritariamente M5 y M15 para barridos, perdiendo la precisión de la entrada en CHoCH/FVG que ocurre inmediatamente después del sweep H1.
- El usuario manual parece ser altamente selectivo, permitiendo un Win Rate del 35% con un ratio R/R que promedia 2:1 a 3:1.

## 4. Por qué el Bot falló donde el Humano ganó
1. **Selección de Horario:** El humano solo opera el "NY Open Killzone". El bot operó todo el día.
2. **Timing de Entrada:** El humano entra en la confirmación 3M; el bot a menudo entraba en el cierre de la vela de sweep, sufriendo stop-outs por "indecisión" del nivel.
3. **Filtro de Ruido:** El humano no toma barridos débiles. El bot tomaba cada sweep de 1 pip.

## 5. Acciones para el Edge Reconstructed
- Limitar la ventana operativa de futuros bots estrictamente a **08:00 - 11:00 NY**.
- Implementar lógica de **3M CHoCH + FVG** como gatillo no negociable.
- Ignorar setups después de las 11:30 NY independientemente de la señal.
