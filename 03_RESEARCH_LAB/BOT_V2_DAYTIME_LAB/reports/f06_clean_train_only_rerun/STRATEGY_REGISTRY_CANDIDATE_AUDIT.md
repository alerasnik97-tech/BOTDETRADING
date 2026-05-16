# STRATEGY REGISTRY CANDIDATE AUDIT

## 1. Status
**STRATEGY_CANDIDATES_READY_FOR_OWNER_DECISION**

## 2. Executive Summary
El motor de ejecución está listo (PREFLIGHT PASS), pero la familia "F06" no existe como tal en el código. Se ha realizado una auditoría estricta de solo lectura sobre `research_lab/strategies/` (95 archivos, 67 exportados en `STRATEGY_REGISTRY`). Se ha identificado que `keltner_volatility_expansion_simple` es el candidato más seguro, limpio y causalmente robusto para representar el concepto de "Volatility Regime" (original de F06), seguido de `campaign3b_session_expansion` para el concepto de expansión de sesión. 

No se ha modificado código, no se han ejecutado backtests y las validaciones/holdouts permanecen intocables. El owner debe tomar las decisiones D1-D5 formales para proceder a la implementación del adapter.

## 3. Registry Inventory Summary
- **Directorio analizado:** `research_lab/strategies/`
- **Total de archivos `.py`:** 95
- **Estrategias registradas en `STRATEGY_REGISTRY`:** 67
- **F06/F08/F12 explícitos:** 0 encontrados. El registro está indexado por `NAME`, no por tag de familia.

## 4. Candidate Scoring Method
Se evaluaron los candidatos bajo las siguientes dimensiones (0-100):
- **Causal Safety:** Ausencia total de lookahead bias (`.shift(-1)`) y correcto uso de variables de estado.
- **Engine Compat:** Compatibilidad de la firma `generate_signal(frame, i, params)` y retorno de diccionario estándar.
- **Phase 3 Fit:** Compatibilidad con EURUSD intradía (07:00-17:00 NY).
- **Simplicity & Overfit Risk:** Minimización de parámetros y umbrales mágicos.
- **F06 Conceptual Fit:** Coherencia con el concepto de volatilidad o regímenes que justificaba originalmente a F06.

## 5. Candidate Table
| Strategy Name | Causal Safety | Engine Compat | Phase 3 Fit | Simplicity | Overfit Risk | F06 Conceptual Fit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `keltner_volatility_expansion_simple` | 100 | 100 | 100 | 90 | Low | High (Volatility Expansion) |
| `campaign3b_session_expansion` | 100 | 100 | 100 | 85 | Low | Medium (Session Overlap) |
| `pm_volatility_squeeze_retest_m5` | 100 | 100 | 80 | 70 | Medium | High (Squeeze/Breakout) |
| `bollinger_mean_reversion_simple` | 100 | 100 | 90 | 90 | Low | Low (Mean Reversion) |
| `ema_trend_pullback` | 100 | 100 | 90 | 90 | Low | Low (Trend Following) |

## 6. Top 5 Shortlist
1. **`keltner_volatility_expansion_simple`** [STRONG CANDIDATE]
   - *Why:* Código extremadamente limpio, causal, pocos parámetros (EMA base, ATR mult). Evalúa compresión seguida de expansión de rango. Es la representación ideal de un "Régimen de Volatilidad".
   - *Suitable as F06:* YES.
2. **`campaign3b_session_expansion`** [STRONG CANDIDATE]
   - *Why:* Lógica causal impecable comparando el rango de Asia vs NY. Ideal para capturar flujos de liquidez intradiarios.
   - *Suitable as F06:* YES (o como F08).
3. **`pm_volatility_squeeze_retest_m5`** [POSSIBLE CANDIDATE]
   - *Why:* Robusta, basada en Bollinger/Keltner squeeze, pero muy orientada a la sesión PM y con bastantes parámetros.
   - *Suitable as F06:* CONDITIONAL.
4. **`bollinger_mean_reversion_simple`** [WEAK CANDIDATE FOR F06]
   - *Why:* Segura y simple, pero no representa el concepto de momentum/breakout que suele asociarse a F06.
   - *Suitable as F06:* NO (Mejor como familia independiente).
5. **`ema_trend_pullback`** [WEAK CANDIDATE FOR F06]
   - *Why:* Demasiado simple (baseline).
   - *Suitable as F06:* NO.

## 7. Recommended F06 Definition
**OPCIÓN 1 — F06 = `keltner_volatility_expansion_simple` + Canonical Config.**
- **Pros:** Transición limpia. Se aprovecha código existente de altísima calidad que ya está en la rama y que conceptualmente reemplaza al viejo F06.
- **Cons:** Requiere que el owner acepte este mapeo explícito.
- **Recommendation:** Es la opción más segura y rápida que preserva la gobernanza.

## 8. Ranking / Config Taxonomy Recommendation
- **Política recomendada:** `SINGLE_CONFIG_FROZEN`.
- **Justificación:** Permitir múltiples configuraciones o sweeps invalida el propósito del "Clean Train-Only Rerun". El objetivo de Phase 3 no es optimizar, sino validar infraestructura con parámetros canónicos inmutables.
- **Config ID recomendado:** `F06_PHASE3_CANONICAL_001`.

## 9. Cost Model Recommendation
- **Política recomendada:** `ftmo` (Lot-based).
- **Escenarios propuestos:**
  - *Base:* `commission_per_lot_round_turn=6.0`, `slippage_pips=0.0`.
  - *Conservative:* `commission_per_lot_round_turn=6.0`, `slippage_pips=0.5` (OBLIGATORIO para preflight realista).
  - *Stress:* `commission_per_lot_round_turn=6.0`, `slippage_pips=1.5`.
- **Decisión del owner requerida:** Validar si se acepta correr el "Clean Rerun" directamente en el escenario `Conservative`.

## 10. Gross R / SL Pips Resolution Recommendation
- **Problema:** El engine no emite `gross_r` ni `sl_pips` en la forma requerida por el ledger final.
- **Recomendación (OPCIÓN A+F combinadas):** El Adapter (fuera del core) debe calcular e inferir el `sl_pips` derivado de la distancia entre `entry_price` y `stop_price` dictados por la señal/trade record del engine, recalculando el desglose de fricciones localmente antes de grabar el CSV maestro.
- **Impacto:** Ninguno. No requiere modificar el Core (UnifiedV7Engine) ni romper esquemas.

## 11. Owner Decisions Required
- **D1:** Definir la estrategia exacta que será F06 (ej: `keltner_volatility_expansion_simple`).
- **D2:** Confirmar si se utilizará un ranking de configuración única (`SINGLE_CONFIG_FROZEN`).
- **D3:** Confirmar el Config ID (`F06_PHASE3_CANONICAL_001`) y congelar sus parámetros (evitando sweeps).
- **D4:** Definir los valores exactos para `spread`, `slippage` y `commission` del Cost Model.
- **D5:** Aprobar la resolución del Adapter para recalcular `gross_r` y `sl_pips` de forma segura sin tocar el core.

## 12. What Remains Forbidden
- Tocar `src/v7_engine/` (Core).
- Correr Backtests o ejecutar estrategias.
- Tocar los años 2025/2026 (Validation/Holdout).
- Usar métricas o resultados contaminados anteriores.
- Hacer sweep de parámetros.

## 13. Final Recommendation
Completar el documento `OWNER_DECISION_D1_D5_TEMPLATE.md` con las elecciones formales. Una vez completado, usar el `NEXT_PROMPT` para validar las decisiones, generar el Adapter e iniciar la ejecución estrictamente controlada.
