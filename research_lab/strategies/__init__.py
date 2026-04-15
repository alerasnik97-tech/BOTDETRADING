from __future__ import annotations

from research_lab.strategies import (
    bollinger_mean_reversion_simple,
    bollinger_mean_reversion_adx_low,
    donchian_breakout_regime,
    ema_trend_pullback,
    keltner_volatility_expansion_simple,
    keltner_squeeze_breakout,
    supertrend_ema_filter,
    strategy_smr,
    strategy_ls_sr,
    strategy_src,
    strategy_vse,
    strategy_ny_br_pure,
    strategy_ny_br_ema,
    strategy_ny_br_mom,
    strategy_sp2_base,
    strategy_sp2_htf_ema,
    strategy_sp2_htf_adx,
)


STRATEGY_REGISTRY = {
    bollinger_mean_reversion_simple.NAME: bollinger_mean_reversion_simple,
    ema_trend_pullback.NAME: ema_trend_pullback,
    bollinger_mean_reversion_adx_low.NAME: bollinger_mean_reversion_adx_low,
    donchian_breakout_regime.NAME: donchian_breakout_regime,
    keltner_volatility_expansion_simple.NAME: keltner_volatility_expansion_simple,
    keltner_squeeze_breakout.NAME: keltner_squeeze_breakout,
    supertrend_ema_filter.NAME: supertrend_ema_filter,
    strategy_smr.NAME: strategy_smr,
    strategy_ls_sr.NAME: strategy_ls_sr,
    strategy_src.NAME: strategy_src,
    strategy_vse.NAME: strategy_vse,
    strategy_ny_br_pure.NAME: strategy_ny_br_pure,
    strategy_ny_br_ema.NAME: strategy_ny_br_ema,
    strategy_ny_br_mom.NAME: strategy_ny_br_mom,
    strategy_sp2_base.NAME: strategy_sp2_base,
    strategy_sp2_htf_ema.NAME: strategy_sp2_htf_ema,
    strategy_sp2_htf_adx.NAME: strategy_sp2_htf_adx,
}
