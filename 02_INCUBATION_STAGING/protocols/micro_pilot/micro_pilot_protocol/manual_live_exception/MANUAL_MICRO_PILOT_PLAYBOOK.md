# Playbook: Micro-Piloto Manual Ultra Chico

**Estado:** `MANUAL_EXCEPTION_ULTRA_SMALL` / `NOT_FULL_INSTITUTIONAL_REAL_APPROVAL`

---

## 1. ¿Qué es este micro-piloto manual?
Es una fase de excepción controlada para ejecutar manualmente la estrategia actual con capital real ultra pequeño. Sirve para validar la ejecución humana, la psicología del operador y la fidelidad de la estrategia en tiempo real mientras el sistema shadow sigue acumulando evidencia estadística.

## 2. ¿Qué NO es?
- No es una aprobación de trading real pleno.
- No es una invitación a improvisar.
- No es un cambio en el core productivo.
- No es el fin de la fase shadow.

## 3. ¿Por qué existe?
Existe tras la validación fuerte del tramo **2026-01-01 a 2026-04-23** (PF 3.06, Expectancy 0.61R). Se autoriza como excepción mientras se espera cumplir el criterio de N >= 10 en la línea shadow (bloqueador actual).

## 4. Estrategia Exacta a Ejecutar
Se opera estrictamente la configuración `tp_1p50_timeout_4h_sl_1p0_longbuf_0p3_win_0_1_mode_close_reclaim_pick_first_levels_all_levels_news_sweep_plus_minus_30m`.

- **Activo:** EURUSD
- **Niveles:** PDH/PDL, Asia H/L, London H/L.
- **Sweep:** Mecha atraviesa nivel y vela cierra por dentro (H1).
- **Confirmación:** Reclaim por cierre de vela M5 dentro de la ventana +0h a +1h del sweep.
- **Entrada:** Apertura de la siguiente vela M5 tras la confirmación.
- **SL:** Extremo del sweep + 1.0 pip de buffer.
- **TP:** 1.5R fijo.
- **Timeout:** 4 horas máximo.

## 5. Límites de Riesgo
- **Riesgo por trade:** 0.10% a 0.25% máximo.
- **Límite diario:** 1 trade por día.
- **Exposición:** Máximo 1 posición abierta.

## 6. Reglas de Entrada
- **Entrar si:** El precio hace sweep claro, confirma en M5 según la regla y el riesgo es >= 2.0 pips.
- **NO entrar si:** Hay duda, el mercado está errático, faltan 30m o menos para noticia de alto impacto, o ya se tomó un trade hoy.

## 7. Reglas Durante la Operación
- No mover el SL a menos que sea para cerrar por timeout.
- No agregar posiciones.
- No cerrar manualmente antes de TP/SL/Timeout por miedo.

## 8. Reglas de Salida
- TP hit (1.5R).
- SL hit (-1.0R).
- Timeout (4h): Cierre a precio de mercado.

## 9. Kill Switch
Si se pierden 3 trades seguidos o el DD supera el 5%, el piloto se detiene inmediatamente.

## 10. Registro y Revisión
- Cada trade debe ir al `trade_journal_template.csv`.
- Cada fin de sesión requiere llenar el `daily_review_template.md`.

---
**SI NO ESTÁ CLARO, NO SE TOMA.**
