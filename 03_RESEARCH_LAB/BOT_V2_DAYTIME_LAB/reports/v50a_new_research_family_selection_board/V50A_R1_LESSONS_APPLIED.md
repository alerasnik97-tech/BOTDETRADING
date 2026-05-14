# V50A R1 LESSONS APPLIED

Para evitar repetir el fracaso de R1, la fase V50A aplica los siguientes filtros de seleccin:

## Reglas de Oro
1. **Gate Combinado Innegociable**: Ninguna familia serǭ evaluada para Full Scope si no pasa simultǭneamente TRAIN (PF >= 1.0) y VAL (PF >= 1.15) en la fase de Gauntlet.
2. **Prioridad a la Causalidad**: Se descarta cualquier familia basada nicamente en "patrones visuales" sin una explicacin de la ineficiencia de mercado subyacente.
3. **MǸtrica N Estricta**: Se requiere un mnimo de trades (N_train >= 30, N_val >= 20) para considerar un resultado como no-aleatorio.
4. **Blindaje de TEST**: El set 2025-2026 estǭ totalmente prohibido para la toma de decisiones de seleccin de familia.
5. **Diferenciacin**: La familia debe operar triggers distintos a Manipulante para evitar la concentracin de riesgo correlacionado.
6. **Anǭlisis de Concentracin**: Se rechazarǭn familias que dependan de un solo mes para obtener rentabilidad en VAL.
