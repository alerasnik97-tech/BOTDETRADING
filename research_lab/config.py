from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


NY_TZ = "America/New_York"
DEFAULT_PAIR = "EURUSD"
DEFAULT_DATA_DIRS = (
    Path("data_free_2020/prepared"),
    Path("data_candidates_2022_2025/prepared"),
)
DEFAULT_HIGH_PRECISION_RAW_DIR = Path("data_precision_raw") / "dukascopy"
DEFAULT_HIGH_PRECISION_PREPARED_DIR = Path("data_precision") / "dukascopy"
DEFAULT_TRADING_ECONOMICS_IMPORT_DIR = Path("data") / "news_imports"
DEFAULT_RAW_NEWS_FILE_OBSOLETE = Path("data/forex_factory_cache.csv")
DEFAULT_NEWS_FILE_OBSOLETE = Path("data/news_eurusd_m15_validated.csv")
DEFAULT_NEWS_V2_UTC_FILE = Path("data/news_eurusd_v2_utc.csv")
DEFAULT_NEWS_ENABLED = False  # FAIL-CLOSED: Reconstruyendo frente News Reliability V2.
DEFAULT_NEWS_SOURCE_APPROVED = False  # RECHAZADO: Usar exclusivamente fuentes UTC validas (Fase 2+).
DEFAULT_RESULTS_DIR = Path("results") / "research_lab_robust"
VISIBLE_CHATGPT_ARCHIVE = Path("000_PARA_CHATGPT.zip")
INITIAL_CAPITAL = 100_000.0
DEFAULT_RISK_PCT = 0.5
DEFAULT_SPREAD_PIPS = 1.2
DEFAULT_SLIPPAGE_PIPS = 0.2
DEFAULT_COMMISSION_ROUNDTURN_USD = 7.0
DEFAULT_PRICE_SOURCE = "bid"
DEFAULT_INTRABAR_EXIT_PRIORITY = "stop_first"
DEFAULT_EXECUTION_MODE = "normal_mode"
DEFAULT_COST_PROFILE = "auto"
DEFAULT_INTRABAR_POLICY = "auto"
SUPPORTED_EXECUTION_MODES = ("normal_mode", "conservative_mode", "high_precision_mode")
SUPPORTED_COST_PROFILES = ("auto", "base", "stress", "precision")
SUPPORTED_INTRABAR_POLICIES = ("auto", "standard", "conservative")
DEFAULT_SEED = 42
DEFAULT_MAX_EVALS_PER_STRATEGY = 8
DEFAULT_WFA_IS_MONTHS = 24
DEFAULT_WFA_OOS_MONTHS = 6
ALT_WFA_IS_MONTHS = 36
ALT_WFA_OOS_MONTHS = 6
MIN_TOTAL_TRADES = 600
MIN_TRADES_PER_MONTH = 1.0
TARGET_TRADES_PER_MONTH_MIN = 15.0
TARGET_TRADES_PER_MONTH_MAX = 25.0

PAIR_META: dict[str, dict[str, Any]] = {
    "EURUSD": {
        "base": "EUR",
        "quote": "USD",
        "pip_size": 0.0001,
        "lot_size": 100_000.0,
        "default_spread_pips": DEFAULT_SPREAD_PIPS,
    },
    "USDJPY": {
        "base": "USD",
        "quote": "JPY",
        "pip_size": 0.01,
        "lot_size": 100_000.0,
        "default_spread_pips": 1.3,
    },
}

STRATEGY_NAMES: tuple[str, ...] = (
    "bollinger_mean_reversion_simple",
    "ema_trend_pullback",
    "bollinger_mean_reversion_adx_low",
    "donchian_breakout_regime",
    "keltner_volatility_expansion_simple",
    "keltner_squeeze_breakout",
    "supertrend_ema_filter",
    "strategy_smr",
    "strategy_ls_sr",
    "strategy_src",
    "strategy_vse",
    "ny_br_pure",
    "ny_br_ema",
    "ny_br_mom",
    "sp2_base",
    "sp2_htf_ema",
    "sp2_htf_adx",
)


SESSION_VARIANTS: dict[str, tuple[str, str]] = {
    "all_day": ("00:00", "23:45"),
    "london_ny": ("03:00", "17:00"),
    "ny_open": ("07:00", "11:00"),
    "light_fixed": ("11:00", "19:00"),
}


@dataclass(frozen=True)
class SessionConfig:
    entry_start: str = "11:00"
    entry_end: str = "19:00"
    force_close: str = "19:00"


@dataclass(frozen=True)
class NewsConfig:
    enabled: bool = DEFAULT_NEWS_ENABLED
    file_path: Path = DEFAULT_NEWS_V2_UTC_FILE
    raw_file_path: Path = DEFAULT_RAW_NEWS_FILE_OBSOLETE
    source_approved: bool = DEFAULT_NEWS_SOURCE_APPROVED
    utc_canonical: bool = True
    pre_minutes: int = 15
    post_minutes: int = 15
    currencies: tuple[str, ...] | None = None
    impact_levels: tuple[str, ...] = ("HIGH",)
    block_new_entries_only: bool = True


@dataclass(frozen=True)
class EngineConfig:
    pair: str = DEFAULT_PAIR
    risk_pct: float = DEFAULT_RISK_PCT
    shock_candle_atr_max: float = 2.2
    assumed_spread_pips: float | None = None
    max_spread_pips: float = DEFAULT_SPREAD_PIPS
    commission_per_lot_roundturn_usd: float = DEFAULT_COMMISSION_ROUNDTURN_USD
    slippage_pips: float = DEFAULT_SLIPPAGE_PIPS
    opening_session_end: str = "13:00"
    late_session_start: str = "17:00"
    high_vol_range_atr: float = 1.0
    spread_opening_multiplier: float = 1.05
    spread_high_vol_multiplier: float = 1.25
    spread_late_session_multiplier: float = 3.0  # HARDENED: 1.1 era un fill irrealista para rollover (17:00 NY)
    slippage_opening_multiplier: float = 1.1
    slippage_high_vol_multiplier: float = 1.5
    slippage_stop_multiplier: float = 1.25
    slippage_target_multiplier: float = 1.0
    slippage_late_session_multiplier: float = 2.0 # HARDENED: penalizar más fuerte rollover slippage
    slippage_forced_close_multiplier: float = 1.1
    slippage_final_close_multiplier: float = 1.05
    stress_spread_multiplier: float = 1.35
    stress_slippage_multiplier: float = 1.6
    ambiguity_slippage_multiplier: float = 1.5
    intrabar_exit_priority: str = DEFAULT_INTRABAR_EXIT_PRIORITY
    max_trades_per_day: int = 2
    max_open_positions: int = 1
    price_source: str = DEFAULT_PRICE_SOURCE
    execution_mode: str = DEFAULT_EXECUTION_MODE
    cost_profile: str = DEFAULT_COST_PROFILE
    intrabar_policy: str = DEFAULT_INTRABAR_POLICY


def time_to_minute(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    return hour * 60 + minute


def resolved_cost_profile(engine_config: EngineConfig) -> str:
    if engine_config.cost_profile != "auto":
        return engine_config.cost_profile
    if engine_config.execution_mode == "conservative_mode":
        return "stress"
    if engine_config.execution_mode == "high_precision_mode":
        return "precision"
    return "base"


def resolved_intrabar_policy(engine_config: EngineConfig) -> str:
    if engine_config.intrabar_policy != "auto":
        return engine_config.intrabar_policy
    return "conservative" if engine_config.execution_mode == "conservative_mode" else "standard"


def with_execution_mode(engine_config: EngineConfig, execution_mode: str) -> EngineConfig:
    normalized = execution_mode.strip().lower()
    if normalized not in SUPPORTED_EXECUTION_MODES:
        raise ValueError(f"Modo de ejecucion no soportado: {execution_mode}")
    cost_profile = engine_config.cost_profile
    intrabar_policy = engine_config.intrabar_policy
    if cost_profile == "auto":
        if normalized == "conservative_mode":
            cost_profile = "stress"
        elif normalized == "high_precision_mode":
            cost_profile = "precision"
        else:
            cost_profile = "base"
    if intrabar_policy == "auto":
        intrabar_policy = "conservative" if normalized == "conservative_mode" else "standard"
    return replace(
        engine_config,
        execution_mode=normalized,
        cost_profile=cost_profile,
        intrabar_policy=intrabar_policy,
    )
