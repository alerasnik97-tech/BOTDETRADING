# DISEÑO ARQUITECTÓNICO DEL BARRIDO DE EXPANSIÓN PARAMÉTRICA (EXPANSION SWEEP DESIGN)

## 1. Filosofía de Búsqueda y Anclaje
El futuro escrutinio en la nube no operará como una cacería ciega de parámetros. Se estructura como una **exploración topográfica de vecindad** anclada de forma estricta y exclusiva en la configuración ganadora certificada:
`cfg_r1_absorption_v4_p3`

- **Presupuesto Dimensional**: Se prohíbe la apertura descontrolada del espacio (ej. 5,000 combinaciones). El barrido inicial queda limitado a un máximo de **100 a 300 configuraciones concurrentes**.

## 2. Inmutabilidad de Restricciones e Higiene
Las siguientes capas operativas se declaran sagradas e inalterables durante el diseño y ejecución del barrido:
- **Activo Único**: `EURUSD`
- **Frecuencia Límite**: `max_trades_per_day = 3`
- **Penalización por Deslizamiento**: Slippage incondicional de `0.2` pips.
- **Modelo de Costos**: Comisiones FTMO activas nativamente.
- **Filtro Macroeconómico**: Impacto y exclusiones de noticias (Data/News) encendidos.
- **Higiene de Truncamiento**: Incidencia nula de cierres de simulación a fin de mes (`EOM = 0`).
- **Blindaje OOS**: Absoluta prohibición de selección, descarte o ajuste utilizando la partición `TEST`. La evaluación de configuraciones candidatas se dirimirá de forma exclusiva sobre las métricas combinadas de `TRAIN` y `VAL`. La muestra `TEST` se reserva para una única corrida de validación final (*single-run final*) sobre el ensamble seleccionado.

## 3. Matriz de Grados de Libertad Permitidos
La variación paramétrica queda estrictamente confinada a las siguientes dimensiones en torno a la semilla ganadora:
1. **Subventana de Apertura**: Alternar entre `08:00-11:00`, `08:30-11:00` y `08:00-10:30` NY.
2. **Fuerza de Rechazo (`wick_to_body_min`)**: Barrido fino entre `2.0` y `3.0` (paso de 0.2).
3. **Ventana de Retorno (`return_inside_max_minutes`)**: Exploración entre `15` y `45` minutos.
4. **Proximidad a Extremos (`rejection_distance_atr_min`)**: Ajustes milimétricos del umbral de cercanía.
5. **Objetivo de Beneficio (Take Profit)**: Rango acotado entre `2.0 R` y `3.0 R`.
6. **Umbral de Break Even (BE)**: Activación entre `+0.8 R` y `+1.2 R`, protegiendo con `+0.2 R` a `+0.5 R`.
7. **Holgura de Parada (SL Buffer)**: Modulación fina del factor ATR extra.

## 4. Veto Explícito de Variables Prohibidas
Queda terminantemente vedado:
- Inyectar nuevas familias lógicas o detectores ajenos a la absorción.
- Incorporar canastas masivas de niveles de soporte/resistencia adicionales.
- Relajar, enmascarar o suprimir los regímenes de costos, slippage, noticias o límites diarios para forzar un abultamiento artificial de las curvas.
