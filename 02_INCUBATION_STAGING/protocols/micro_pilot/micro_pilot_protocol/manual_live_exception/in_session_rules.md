# Reglas Intra-Sesión: Ejecución y Gestión

Reglas innegociables durante la ventana operativa.

## 1. Validación de Entrada (Filtro Duro)
- [ ] **Sweep:** ¿El precio atravesó un nivel H1 y cerró por dentro de forma evidente?
- [ ] **Confirmación M5:** ¿Hay una vela M5 que cierre de vuelta sobre el nivel (long) o bajo el nivel (short) dentro de los 60m posteriores al sweep?
- [ ] **Buffer:** ¿He aplicado el buffer de entrada (0.3 pips en long, 0.0 en short)?
- [ ] **Riesgo:** ¿El stop loss está a una distancia mínima de 2.0 pips?

## 2. Colocación de Órdenes
- El SL debe ir **exactamente** en el extremo del sweep más 1.0 pip de buffer.
- El TP debe ser **exactamente** 1.5 veces el riesgo asumido (distancia Entry-SL).
- **PROHIBIDO** ampliar el SL por miedo a ser sacado.

## 3. Gestión de la Posición
- **No Intervención:** Una vez abierta la posición, no se toca hasta que toque TP o SL.
- **Excepción Timeout:** Si pasan 4 horas y el trade sigue abierto, se cierra a mercado inmediatamente.
- **Noticias:** Si una noticia de alto impacto ocurre mientras el trade está abierto, se mantiene el plan (el filtro debió evitar la entrada si estaba cerca).

## 4. Disciplina de Ejecución
- Si el precio se aleja demasiado sin confirmar, el setup se invalida.
- Si hay dudas sobre si el cierre fue "por dentro" o "por fuera" del nivel, **NO SE ENTRA**.
- Solo 1 operación a la vez.

---
**EL ÉXITO ES SEGUIR EL PROCESO, NO EL RESULTADO DEL TRADE.**
