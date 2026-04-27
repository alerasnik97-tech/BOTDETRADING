# Strategy Master Matrix - EUR/USD PM Lab

| Name | Origin | Family | TF Req | Status | OOS PF | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `zscore_mean_reversion_pm` | Gemini | Stat Rev | M15 | **Audit Failed**| 0.94 | Edge nominal (1.00) colapsa sin news filter. |
| `ict_fvg_liquidity_gap` | Gemini | Price Action | M15 | **Audit Failed**| 0.68 | Demasiado ruido intradía. |
| `ict_silver_bullet_pm` | Gemini | Time | M15 | Ready | < 1.0 | Requiere ajuste de ventana interna. |
| `triple_macd_filter` | Gemini | Trend | M15 | Repaired | - | Re-evaluación en curso. |
| `larry_connors_rsi2` | Gemini | Mean Rev | M15 | Repaired | - | Re-evaluación en curso. |
| `turtle_soup_fade` | Gemini | Liquidity | M15 | Low Edge | < 1.0 | Candidata a tuning de lookback. |
| `nr7_breakout` | Gemini | Volatility | M15 | Low Edge | < 1.0 | Muy restrictiva. |

## Wave 5 — Liquidity Bracketing Research

| Name | Origin | Family | TF Req | Status | OOS PF | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `london_sweep_reversion_pm`| Research | Liquidity | M15 | **Rejected** | 0.61 | Falsas rupturas dominan el régimen. |
| `asia_sweep_reversion_pm` | Research | Liquidity | M15 | **Rejected** | 0.55 | El rango de Asia es irrelevante para la PM. |
| `prev_day_extrema_sweep` | Research | Liquidity | M15 | **Rejected** | 0.69 | Nivel más respetado, pero sin edge real. |

---

## Wave 4 — HTF Regime Research

## Wave 3 — Incubator (Paralelo)

## Categorization Key
- **Ready:** Implementada y validada en backtest masivo.
- **Incubación:** Código base listo en `/incubator/` (sin registro activo).
- **Repaired:** Correcciones técnicas realizadas, pendiente de nueva corrida.
- **Líder:** Candidata prioritaria para optimización de parámetros.
