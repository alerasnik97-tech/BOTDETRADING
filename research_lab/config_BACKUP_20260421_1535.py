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
DEFAULT_NEWS_ENABLED = True  # ENABLED for research phase
DEFAULT_NEWS_SOURCE_APPROVED = True  # APPROVED for UTC-based sources
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
MIN_TOTAL_TRADES = 100  # Relajado para fase de descubrimiento PM
MIN_TRADES_PER_MONTH = 0.5  # Minimo 1 trade cada 2 meses para investigacion
TARGET_TRADES_PER_MONTH_MIN = 3.0
TARGET_TRADES_PER_MONTH_MAX = 15.0

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

PAIR_CANONICAL_DATA_DIRS: dict[str, tuple[Path, ...]] = {
    "EURUSD": DEFAULT_DATA_DIRS,
    "USDJPY": (
        Path("data_usdjpy_2016_2021") / "prepared",
        Path("data_usdjpy_2022_2025") / "prepared",
    ),
}

PAIR_CANONICAL_NEWS_FILES: dict[str, Path] = {
    "EURUSD": Path("data") / "news_eurusd_am_fortress_v3.csv",
    "USDJPY": Path("data") / "news_usdjpy_fortress_v1.csv",
}

PAIR_CANONICAL_NEWS_SUMMARY_FILES: dict[str, Path] = {
    "EURUSD": Path("data") / "news_eurusd_am_fortress_v3_summary.json",
    "USDJPY": Path("data") / "news_usdjpy_fortress_v1_summary.json",
}

PAIR_DEFAULT_NEWS_IMPACTS: dict[str, tuple[str, ...]] = {
    "EURUSD": ("HIGH",),
    "USDJPY": ("HIGH",),
}

PAIR_FIRST_FAMILY_HIGH_PRECISION_REQUIRED: dict[str, bool] = {
    "EURUSD": True,
    "USDJPY": False,
}

STRATEGY_NAMES: tuple[str, ...] = (
    "prev_day_sweep_reversion_pm",
    "asia_london_sweep_reversion_pm",
    "midday_false_break_range",
    "midday_range_breakout_continuation",
    "h1_trend_pullback_pm",
    "adr_exhaustion_fade",
    "h1_inside_bar_break_pm",
    "daily_open_mean_reversion_pm",
    "ict_silver_bullet_pm",
    "zscore_mean_reversion_pm",
    "donchian_intraday_breakout",
    "adx_momentum_breakout",
    "turtle_soup_fade",
    "nr7_breakout",
    "triple_macd_filter",
    "ema_alignment_9_21_50",
    "larry_connors_rsi2",
    "ict_fvg_liquidity_gap",
    "h1_aligned_fvg",
    "h1_trend_pullback_v2",
    "h1_gated_zscore",
    "london_sweep_reversion_pm",
    "asia_sweep_reversion_pm",
    "prev_day_extrema_sweep",
    "am_silver_bullet_ny",
    "eurusd_h1_liquidity_sweep_m15",
    "eurusd_am_post_news_external_liquidity_shift",
    "campaign3_extended_session_sweep",
    "campaign3_midday_daily_reclaim",
    "campaign3_post_news_continuation",
    "campaign3_afternoon_compression_breakout",
    "campaign3_london_ny_hybrid",
    "campaign3_late_session_momentum",
    "campaign3_mtf_alignment",
    "campaign3b_midday_reclaim",
    "campaign3b_compression_breakout",
    "campaign3b_post_news_continuation",
    "campaign3b_session_expansion",
)

SESSION_VARIANTS: dict[str, tuple[str, str]] = {
    "all_day": ("00:00", "23:45"),
    "london_ny": ("03:00", "17:00"),
    "ny_open": ("07:00", "11:00"),
    "research_08_1630": ("08:00", "16:30"),
    "light_fixed": ("11:00", "19:00"),
    "pm_11_12": ("11:00", "12:00"),
    "pm_12_1330": ("12:00", "13:30"),
    "pm_1330_16": ("13:30", "16:00"),
    "pm_1630_19": ("16:30", "19:00"),
    "pm_11_1630": ("11:00", "16:30"),
    "pm_silver_bullet": ("14:00", "15:00"),
    "pm_11_16": ("11:00", "16:00"),
    "pm_11_17": ("11:00", "17:00"),
    "am_08_11": ("08:00", "11:00"),
    "asia_19_03": ("19:00", "03:00"),
    "london_03_07": ("03:00", "07:00"),
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
    pre_minutes: int = 30
    post_minutes: int = 60
    currencies: tuple[str, ...] | None = None
    impact_levels: tuple[str, ...] = ("HIGH",)
    block_new_entries_only: bool = False  # FORTRESS: Default to full block
    fomc_only: bool = False
    forced_exit_pre_news: bool = True     # FORTRESS: Kill positions before impact
    cancel_pending_pre_news: bool = True  # FORTRESS: Kill signals before impact
    pre_news_exit_minutes: int = 10       # 10m buffer for liquidity exit


@dataclass(frozen=True)
class EngineConfig:
    pair: str = DEFAULT_PAIR
    risk_pct: float = DEFAULT_RISK_PCT
    shock_candle_atr_max: float = 4.0
    assumed_spread_pips: float | None = None
    max_spread_pips: float = 3.0
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
    session_cutoff: str | None = None
    enforce_hard_stop: bool = True        # FORTRESS: Reject signals without SL


def pair_currencies(pair: str) -> tuple[str, str]:
    meta = PAIR_META[pair]
    return str(meta["base"]), str(meta["quote"])


def canonical_prepared_data_dirs(pair: str) -> tuple[Path, ...]:
    try:
        return PAIR_CANONICAL_DATA_DIRS[pair]
    except KeyError as exc:
        raise ValueError(f"No hay contrato canonico de datos para {pair}") from exc


def canonical_news_file(pair: str) -> Path:
    try:
        return PAIR_CANONICAL_NEWS_FILES[pair]
    except KeyError as exc:
        raise ValueError(f"No hay dataset canonico de noticias para {pair}") from exc


def canonical_news_summary_file(pair: str) -> Path:
    try:
        return PAIR_CANONICAL_NEWS_SUMMARY_FILES[pair]
    except KeyError as exc:
        raise ValueError(f"No hay summary canonico de noticias para {pair}") from exc


def default_news_impacts(pair: str) -> tuple[str, ...]:
    try:
        return PAIR_DEFAULT_NEWS_IMPACTS[pair]
    except KeyError as exc:
        raise ValueError(f"No hay contrato de impactos por defecto para {pair}") from exc


def first_family_requires_high_precision(pair: str) -> bool:
    try:
        return bool(PAIR_FIRST_FAMILY_HIGH_PRECISION_REQUIRED[pair])
    except KeyError as exc:
        raise ValueError(f"No hay politica de precision declarada para {pair}") from exc


def canonical_news_config(
    pair: str,
    *,
    enabled: bool = True,
    pre_minutes: int = 30,
    post_minutes: int = 60,
    forced_exit_pre_news: bool = True,
    cancel_pending_pre_news: bool = True,
    pre_news_exit_minutes: int = 10,
) -> NewsConfig:
    return NewsConfig(
        enabled=enabled,
        file_path=canonical_news_file(pair),
        raw_file_path=DEFAULT_RAW_NEWS_FILE_OBSOLETE,
        source_approved=True,
        currencies=tuple(sorted(pair_currencies(pair))),
        impact_levels=default_news_impacts(pair),
        pre_minutes=pre_minutes,
        post_minutes=post_minutes,
        forced_exit_pre_news=forced_exit_pre_news,
        cancel_pending_pre_news=cancel_pending_pre_news,
        pre_news_exit_minutes=pre_news_exit_minutes,
    )


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
