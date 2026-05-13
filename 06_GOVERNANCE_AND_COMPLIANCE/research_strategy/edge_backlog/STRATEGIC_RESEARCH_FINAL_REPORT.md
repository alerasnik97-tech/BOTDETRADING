## Estado
EDGE_BACKLOG_READY

## Resumen
Se ha constituido de forma aséptica y profesional el **Backlog de Inteligencia Estratégica Institucional** en paralelo, fundamentado en la disección forense del núcleo discrecional del usuario (“Barrido de liquidez + cambio de estructura”) y el modelado físico de fricción sobre las traducciones fallidas previas. Toda la documentación reside intacta y auditada bajo la ruta autorizada de gobierno:  
`C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\06_GOVERNANCE_AND_COMPLIANCE\research_strategy\edge_backlog\`

## Qué aprendimos de M2/M3
*   **Dominancia Negativa del Costo Real:** Manipulante 3 colapsó incondicionalmente en validación al someterse a fricción asimétrica realista; un Profit Factor idealizado sin costos ($\text{PF} = 0.8181$ con `slippage = 0.0`) carece de viabilidad matemática de alcanzar el estándar mínimo de aprobación ($\text{PF}_{\text{net}} > 1.15$) al inyectar markdowns de $0.2$ pips.
*   **Peligro de Minería de Datos (Overfitting):** Manipulante 2 falló por sobreoptimizar combinaciones estáticas en marcos amplios dentro de la muestra, tomando ruido aleatorio de baja volatilidad en horas de la tarde como entradas válidas.
*   **Ruido por Barridos Superficiales:** Programar una simple superación booleana de un máximo/mínimo absoluto sin acotar la profundidad física de la penetración introduce un ruido abrumador proveniente de simples ensanchamientos transitorios del spread.
*   **Necesidad de Causalidad Estricta:** Se corroboró la asimilación exitosa del motor V7 "Next-Bar-Execution" eliminando definitivamente los sesgos históricos de espionaje de velas futuras (Look-Ahead Bias de la Fase 19).

## Qué falta capturar de la lógica manual
*   **Selectividad Intradiaria Extrema:** El usuario restringe sus 841 trades empíricos (con un sorprendente Profit Factor de 1.88) rigurosamente a la ventana de máxima liquidez interbancaria (**08:00 a 11:00 NY Killzone**), ignorando por completo el resto del día.
*   **Calibración Visual del Quiebre en 3M:** El humano opera en temporalidad de **3M** exigiendo un desplazamiento posterior masivo y agresivo (velas cerrando contundentemente por encima del último fractal de absorción real). El bot a menudo ha disparado órdenes en micro-quiebres indecisos.
*   **Contexto de Flujo y Cancelación:** El ojo humano evalúa la intencionalidad direccional previa y aplica una cancelación mental si el precio se escapa directo al objetivo teórico de TP sin retroceder a llenar el FVG (*mitigación fallida*), evitando entrar tarde en un impulso ya agotado.

## Top 5 hipótesis futuras
1.  **Sweep Quality + Displacement Gate (`HYP_001`):** Condiciona el *edge* a que el barrido tenga una profundidad física acotada ($1.5$ a $12$ pips) e inmediatamente detone una expansión violenta del ATR de las velas de reversión en 3M confirmando volumen atrapado.
2.  **Post-News Liquidity Reversal (`HYP_002`):** Emplea las publicaciones Tier-1 del calendario premium curado `news_eurusd_am_fortress_v3.csv` como el *catalizador exclusivo* para capturar reversiones post-shock tras una ventana de absorción de 60 a 180 segundos.
3.  **London Sweep into NY Reversal (`HYP_003`):** Acopla el flujo interbancario exigiendo que la apertura de Nueva York barra el extremo absoluto fijado por la sesión de Londres para orquestar una reversión sostenida durante el Killzone.
4.  **Previous Day Liquidity Reclaim (`HYP_004`):** Rastrea el fracaso de los quiebres de PDH/PDL exigiendo un cierre de vela físico de regreso al rango diario para confirmar la trampa institucional y apuntar a la liquidez interna.
5.  **Failed Breakout with Volatility Compression (`HYP_007`):** Inyecta un filtro ortogonal de régimen de volatilidad macro ($\text{ATR}_{\text{daily}} < 50$ pips) para habilitar reversiones exclusivamente en jornadas de compresión lateral propensas a oscilar de extremo a extremo.

## Recomendación
- **esperar resultado de MANIPULANTE 4:** Mantener congelada la asignación de recursos y **esperar de forma incondicional el desenlace y dictamen de la corrida activa de MANIPULANTE 4** orquestada por el Agente 1 (Research).
- **si M4 falla, siguiente hipótesis recomendada:** Promover de forma prioritaria y aséptica el desarrollo algorítmico de la hipótesis **Sweep Quality + Displacement Gate (`HYP_001`)** acotando el barrido dimensional a un máximo estricto de 12 configuraciones.
- **si M4 funciona, cómo escalar sin data mining:** Si Manipulante 4 logra aislar un candidato viable, prohibir la adición de sub-filtros booleanos post-hoc. Exigir la validación del edge bajo una prueba continua obligatoria de **0.2 pips de slippage asimétrico** superando un $\text{PF}_{\text{val\_net}} \ge 1.15$, verificando la supervivencia en la partición testigo TEST y confirmando la coincidencia milimétrica mediante auditoría independiente de código.

## Prohibiciones respetadas
Se certifica de manera categórica el acatamiento absoluto de las pautas de confinamiento del Agente Paralelo:
- **no código tocado:** Confirmado (Cero modificaciones en archivos `.py`, motores o binarios del repositorio).
- **no datos tocados:** Confirmado (Bóvedas de mercado de ticks y vectores de noticias preservados en solo lectura).
- **no runner tocado:** Confirmado (Orquestadores intradiarios de simulación sin alterar).
- **no tests tocados:** Confirmado (Árbol de integración y pruebas unitarias inmutable).
- **no ZIP tocado:** Confirmado (Empaquetado oficial `000_PARA_CHATGPT.zip` inalterado).
- **no backtest:** Confirmado (Cero invocaciones a simulaciones intradiarias o consumo de I/O de Parquet).
- **no sweep:** Confirmado (Cero barridos dimensionales o minería de hiperparámetros).
- **no commit:** Confirmado (Cero confirmaciones de cambios locales en el árbol de trabajo).
- **no push:** Confirmado (Cero envíos o sincronizaciones con servidores remotos de control de versiones).
- **no Explorer:** Confirmado (Cero aperturas de interfaces gráficas del sistema operativo).
