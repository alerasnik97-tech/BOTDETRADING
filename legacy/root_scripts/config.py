from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


NY_TZ = "America/New_York"
DEFAULT_PAIR = "EURUSD"
DEFAULT_DATA_DIRS = (
    Path("data_free_2020/prepared"),
    Path("data_candidates_2022_2025/prepared"),
)
DEFAULT_NEWS_FILE = Path("data/forex_factory_cache.csv")
DEFAULT_RESULTS_DIR = Path("results")
INITIAL_CAPITAL = 100_000.0
SLIPPAGE_PIPS = 0.2

PAIR_META: dict[str, dict[str, Any]] = {
    "EURUSD": {
        "base": "EUR",
        "quote": "USD",
        "pip_size": 0.0001,
        "default_spread_pips": 0.7,
    },
    "USDJPY": {
        "base": "USD",
        "quote": "JPY",
        "pip_size": 0.01,
        "default_spread_pips": 0.9,
    },
}

MODEL_MODES: tuple[str, ...] = ("hybrid", "range_only", "breakout_only")

# 6 packs x 3 modos = 18 combinaciones por defecto.
MODEL_VARIANTS: tuple[
    tuple[int, int, float, int, float, bool, int, float, float, bool, bool, str, float, float, int, float],
    ...,
] = (
    (16, 3, 0.5, 9, 2.0, False, 40, 60, 1.0, False, True, "atr_stop", 1.0, 1.5, 3, 1.0),
    (18, 5, 0.8, 9, 2.0, True, 35, 65, 1.0, False, True, "atr_stop", 1.2, 1.8, 3, 1.5),
    (20, 8, 1.0, 14, 2.2, True, 30, 70, 1.2, True, False, "compression_stop", 1.0, 1.8, 6, 1.5),
    (18, 5, 0.8, 14, 2.2, False, 35, 65, 1.2, True, True, "compression_stop", 1.2, 1.5, 6, 2.0),
    (16, 3, 0.5, 14, 2.0, True, 40, 60, 1.0, False, False, "atr_stop", 1.0, 1.8, 3, 2.0),
    (20, 8, 1.0, 9, 2.2, False, 30, 70, 1.2, True, True, "compression_stop", 1.2, 1.8, 6, 1.0),
)


@dataclass(frozen=True)
class SessionConfig:
    force_close: str = "19:00"
    entry_start: str = "11:00"
    entry_end: str = "17:30"
    entry_cutoff_minutes: int = 45


@dataclass(frozen=True)
class NewsConfig:
    enabled: bool = True
    file_path: Path = DEFAULT_NEWS_FILE
    pre_minutes: int = 15
    post_minutes: int = 15


@dataclass(frozen=True)
class StrategyParams:
    pair: str = DEFAULT_PAIR
    risk_pct: float = 0.5
    model_mode: str = "hybrid"
    adx_trend_min: int = 18
    ema200_slope_lookback: int = 5
    trend_progress_atr_min: float = 0.8
    ema_distance_atr_min: float = 0.10
    bb_period: int = 20
    bb_std: float = 2.0
    range_rsi_period: int = 9
    range_rsi_low: float = 35.0
    range_rsi_high: float = 65.0
    range_signal_candle_atr_max: float = 1.5
    range_stop_atr: float = 1.0
    range_target_rr: float = 1.2
    range_be_enabled: bool = False
    compression_bars: int = 6
    compression_atr_mult: float = 1.0
    breakout_candle_atr_max: float = 1.5
    breakout_stop_mode: str = "atr_stop"
    breakout_stop_atr: float = 1.0
    breakout_target_rr: float = 1.5
    breakout_be_enabled: bool = False
    breakout_enabled: bool = True
    cooldown_bars: int = 3
    daily_loss_limit_r: float = 1.5
    shock_candle_atr_max: float = 2.2
    max_spread_pips: float = 1.2
    max_trades_per_day: int = 2
    max_trades_per_direction: int = 1
    require_breakout_above_below_ema20: bool = True

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pair"] = self.pair.upper().strip()
        return payload


def optimization_grid(pair: str, risk_pct: float, max_combinations: int = 18) -> list[tuple[StrategyParams, NewsConfig]]:
    candidates: list[tuple[StrategyParams, NewsConfig]] = []
    session = SessionConfig()
    for model_mode in MODEL_MODES:
        for (
            adx_trend_min,
            ema200_slope_lookback,
            trend_progress_atr_min,
            range_rsi_period,
            bb_std,
            range_be_enabled,
            range_rsi_low,
            range_rsi_high,
            range_stop_atr,
            breakout_be_enabled,
            breakout_enabled,
            breakout_stop_mode,
            breakout_stop_atr,
            breakout_target_rr,
            cooldown_bars,
            daily_loss_limit_r,
        ) in MODEL_VARIANTS:
            params = StrategyParams(
                pair=pair,
                risk_pct=risk_pct,
                model_mode=model_mode,
                adx_trend_min=adx_trend_min,
                ema200_slope_lookback=ema200_slope_lookback,
                trend_progress_atr_min=trend_progress_atr_min,
                ema_distance_atr_min=0.10 if trend_progress_atr_min <= 0.5 else 0.15,
                bb_period=20,
                bb_std=bb_std,
                range_rsi_period=range_rsi_period,
                range_rsi_low=range_rsi_low,
                range_rsi_high=range_rsi_high,
                range_signal_candle_atr_max=1.5 if range_be_enabled else 1.8,
                range_stop_atr=range_stop_atr,
                range_target_rr=1.2 if bb_std <= 2.0 else 1.5,
                range_be_enabled=range_be_enabled,
                compression_bars=4 if ema200_slope_lookback == 3 else 6,
                compression_atr_mult=0.8 if breakout_stop_mode == "atr_stop" else 1.0,
                breakout_candle_atr_max=1.2 if breakout_stop_mode == "compression_stop" else 1.5,
                breakout_stop_mode=breakout_stop_mode,
                breakout_stop_atr=breakout_stop_atr,
                breakout_target_rr=breakout_target_rr,
                breakout_be_enabled=breakout_be_enabled,
                breakout_enabled=(breakout_enabled if model_mode != "range_only" else False),
                cooldown_bars=cooldown_bars,
                daily_loss_limit_r=daily_loss_limit_r,
            )
            news_cfg = NewsConfig(
                enabled=True,
                file_path=DEFAULT_NEWS_FILE,
                pre_minutes=15,
                post_minutes=15,
            )
            candidates.append((params, news_cfg))
            if len(candidates) >= max_combinations:
                return candidates
    return candidates
