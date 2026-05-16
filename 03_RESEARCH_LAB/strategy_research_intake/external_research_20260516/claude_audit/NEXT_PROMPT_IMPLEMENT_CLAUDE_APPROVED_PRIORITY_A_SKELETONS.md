# NEXT PROMPT: Implement Claude-Approved Priority A Strategy Skeletons

**Status**: READY FOR EXECUTION
**Approved by**: Claude Opus 4.7 Audit (2026-05-16)
**Strategies**: 4 (NOT 5 — ED-01 removed due to uncertified news data)

---

## STRICT RULES (APPLY TO ALL IMPLEMENTATION)

1. **TRAIN-ONLY**: All code operates on train data only. NO access to holdout (2025-2026).
2. **NO BACKTEST**: Skeleton = signal generation + logging. No PnL, no equity curve, no performance metrics.
3. **NO OPTIMIZATION**: Parameters are FIXED from source research. No grid search, no fitting.
4. **NO ENGINE MODIFICATION**: Do not touch `strategies/`, `engine/`, `data/`, or any production code.
5. **NO DATA MODIFICATION**: Do not alter, resample, or filter existing data files.
6. **RESEARCH_LAB ONLY**: All output goes in `03_RESEARCH_LAB/strategy_skeletons/`.
7. **GIT DISCIPLINE**: Explicit staging (name each file). No `git add .`. No force push. Branch per task.
8. **MAX 3 TRADES/DAY per strategy**: Hard limit in skeleton logic.
9. **spread_ok FILTER**: Every entry checks spread < threshold (0.5 pips default).
10. **news_ok FILTER**: Every entry excludes +-5min around known events (use placeholder until news data certified).

---

## STRATEGIES TO IMPLEMENT (4 total)

### Skeleton 1: MR-01 — Anchor Elastic

**File**: `03_RESEARCH_LAB/strategy_skeletons/skeleton_mr01_anchor_elastic.py`

**Logic**:
- Compute APM (VWAP anchored to 07:00 NY) from M1 OHLCV
- Compute deviation = abs(close - APM) / ATR(14)
- IF deviation > 1.2 AND ADX(14) < 22 AND spread_ok AND news_ok:
  - Mark signal direction (LONG if below APM, SHORT if above)
  - Entry: close of M1 candle re-entering the 1.2 ATR band
  - TP: APM level (max 1.5R)
  - SL: extreme of excursion + 0.5*ATR buffer
- Max 1 entry per day
- Log: timestamp, direction, entry_price, tp, sl, apm_value, deviation_atr, adx_value

**Parameters** (FIXED, do not optimize):
- atr_period: 14
- adx_period: 14
- deviation_threshold_atr: 1.2
- adx_max: 22
- tp_max_r: 1.5
- sl_buffer_atr: 0.5

**Window**: 09:30-15:30 NY

---

### Skeleton 2: VE-01 — RV Shock Break

**File**: `03_RESEARCH_LAB/strategy_skeletons/skeleton_ve01_rv_shock_break.py`

**Logic**:
- Compute rv5 = realized vol from last 5 M1 returns (std * sqrt(252*78))
- Compute rv15 = realized vol from last 15 M1 returns
- Compute rv15_percentile = percentile rank of current rv15 vs rolling 20-day rv15 values
- Compute Donchian high/low over last 30 M5 bars
- IF rv15_percentile <= 30 (compression phase):
  - Watch for breakout: close_M5 > donchian_high OR close_M5 < donchian_low
  - IF breakout AND rv5 >= 2 * median(rv5, 20days) AND spread_ok AND news_ok:
    - Entry: close of breakout M5 bar
    - Direction: LONG if break above, SHORT if break below
    - TP: 2.0R
    - SL: opposite side of breakout bar (low for LONG, high for SHORT)
- Max 1 entry per day
- Log: timestamp, direction, entry, tp, sl, rv5, rv15, rv15_pct, donchian_h, donchian_l

**Parameters** (FIXED):
- rv5_window: 5 (M1 bars)
- rv15_window: 15 (M1 bars)
- rv15_compression_pct: 30
- donchian_period: 30 (M5 bars)
- rv5_shock_multiple: 2.0
- tp_r: 2.0

**Window**: 07:30-11:00 NY

---

### Skeleton 3: TP-01 — Trend Day EMA Pullback

**File**: `03_RESEARCH_LAB/strategy_skeletons/skeleton_tp01_trend_day_ema_pullback.py`

**Logic**:
- At 09:30 NY, qualify as Trend Day:
  - price_range_0700_0930 > 1.5 * ATR(14, daily)
  - Close at 09:30 within 20% of session high (uptrend) or low (downtrend)
  - ADX(14, M15) > 25
- IF Trend Day qualified:
  - Wait for first pullback touching EMA20(M5) without closing below EMA50(M5)
  - Entry: close of M1 candle confirming reversal in trend direction
  - TP: 2.0R or trailing stop below EMA20
  - SL: below EMA50 (for LONG) or above EMA50 (for SHORT)
  - Filters: spread_ok AND news_ok
- Max 1 entry per day (first pullback only)
- Log: timestamp, direction, entry, tp, sl, trend_day_score, ema20, ema50, adx

**Parameters** (FIXED):
- ema_fast: 20
- ema_slow: 50
- adx_period: 14
- adx_min: 25
- trend_day_atr_multiple: 1.5
- tp_r: 2.0
- trailing: EMA20

**Window**: 09:30-13:30 NY (entry window after Trend Day qualification)

---

### Skeleton 4: SD-01 — Europe Extreme Failure

**File**: `03_RESEARCH_LAB/strategy_skeletons/skeleton_sd01_europe_extreme_failure.py`

**Logic**:
- Compute European session range: high/low from 02:00-07:00 NY
- Compute session_midpoint = (eu_high + eu_low) / 2
- After 07:00 NY, watch for false extension:
  - Price exceeds eu_high (or falls below eu_low) by < 0.08 * ATR(14)
  - Then re-enters the European range within 3 M1 candles
- IF false extension detected AND spread_ok AND news_ok:
  - Entry: close of re-entry candle
  - Direction: SHORT if failed break above, LONG if failed break below
  - TP: session_midpoint
  - SL: extreme of the false extension + 0.3*ATR buffer
- Max 1 entry per day
- Log: timestamp, direction, entry, tp, sl, eu_high, eu_low, extension_pips, reentry_bars

**Parameters** (FIXED):
- eu_session_start: 02:00 NY
- eu_session_end: 07:00 NY
- max_extension_atr: 0.08
- max_reentry_bars: 3
- sl_buffer_atr: 0.3
- tp_target: session_midpoint

**Window**: 07:00-11:00 NY

---

## OUTPUT FORMAT (each skeleton)

```python
"""
Strategy Skeleton: {ID} — {Name}
Status: SKELETON_ONLY (no backtest, no optimization)
Audit: Claude Opus 4.7, 2026-05-16
Branch: governance/claude-strategy-intake-audit-20260516
"""

class Skeleton_{ID}:
    # Fixed parameters from research (DO NOT OPTIMIZE)
    PARAMS = { ... }
    
    def generate_signals(self, ohlcv_m1, ohlcv_m5, date):
        """Return list of Signal objects for given date. Max 1 per day."""
        ...
    
    def check_filters(self, timestamp, spread):
        """Return True if spread_ok AND news_ok."""
        ...

# Signal = namedtuple('Signal', ['timestamp', 'direction', 'entry', 'tp', 'sl', 'metadata'])
```

---

## WHAT THIS PROMPT DOES NOT AUTHORIZE

- NO backtest execution
- NO PnL calculation
- NO parameter optimization
- NO holdout data access
- NO engine integration
- NO git add . (explicit file staging only)
- NO ED-01 implementation (news data uncertified)

---

## VALIDATION AFTER IMPLEMENTATION

After skeletons are written, verify:
1. Each file is importable (`python -c "import skeleton_mr01_anchor_elastic"`)
2. No hardcoded data paths (parameters only)
3. No backtest logic embedded
4. No access to 2025/2026 data
5. spread_ok and news_ok filters present in every entry path
6. Max 1 trade/day enforced

---
Generated: 2026-05-16 by Claude Opus 4.7
Replaces: STRATEGY_SKELETON_PROMPT.md (Gemini's version included ED-01 — governance violation)
