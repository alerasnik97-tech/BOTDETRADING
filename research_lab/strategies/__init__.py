from __future__ import annotations

from research_lab.strategies import (
    bollinger_mean_reversion_simple,
    bollinger_mean_reversion_adx_low,
    donchian_breakout_regime,
    ema_trend_pullback,
    keltner_volatility_expansion_simple,
    keltner_squeeze_breakout,
    supertrend_ema_filter,
)


STRATEGY_REGISTRY = {
    bollinger_mean_reversion_simple.NAME: bollinger_mean_reversion_simple,
    ema_trend_pullback.NAME: ema_trend_pullback,
    bollinger_mean_reversion_adx_low.NAME: bollinger_mean_reversion_adx_low,
    donchian_breakout_regime.NAME: donchian_breakout_regime,
    keltner_volatility_expansion_simple.NAME: keltner_volatility_expansion_simple,
    keltner_squeeze_breakout.NAME: keltner_squeeze_breakout,
    supertrend_ema_filter.NAME: supertrend_ema_filter,
}
