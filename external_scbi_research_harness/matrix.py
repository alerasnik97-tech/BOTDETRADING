from __future__ import annotations

from dataclasses import replace
from itertools import product

from .config import DEFAULT_BODY_STRENGTH_THRESHOLD, TruthModelConfig

TP_VALUES = (1.0, 1.25, 1.5, 1.75, 2.0)
TIMEOUT_VALUES = (2, 4, 6)
SL_BUFFER_VALUES = (0.5, 1.0, 1.5)
LONG_ENTRY_BUFFER_VALUES = (0.0, 0.3, 0.5)
CONFIRMATION_WINDOWS = ((0, 1), (1, 2), (0, 2))
CONFIRMATION_MODES = ("close_reclaim", "close_reclaim_body_strength")
CONFIRMATION_PICKS = ("first", "best")
LEVEL_PROFILES = ("all_levels", "pd_only", "asia_only", "london_only")
NEWS_MODES = ("none", "sweep_plus_minus_30m", "sweep_plus_minus_60m", "post_news_cooldown_60m")


def _variant_id(config: TruthModelConfig) -> str:
    return (
        f"tp_{config.tp_r:.2f}_timeout_{config.timeout_hours}h"
        f"_sl_{config.sl_buffer_pips:.1f}"
        f"_longbuf_{config.long_entry_buffer_pips:.1f}"
        f"_win_{config.confirmation_window_start_hours}_{config.confirmation_window_end_hours}"
        f"_mode_{config.confirmation_mode}"
        f"_pick_{config.confirmation_pick}"
        f"_levels_{config.level_profile}"
        f"_news_{config.news_mode}"
    ).replace(".", "p")


def build_baseline_config(start_date: str, end_date: str) -> TruthModelConfig:
    return TruthModelConfig(
        variant_id="baseline_truth_model",
        profile_name="baseline",
        start_date=start_date,
        end_date=end_date,
        tp_r=1.5,
        timeout_hours=4,
        sl_buffer_pips=1.0,
        long_entry_buffer_pips=0.3,
        short_entry_buffer_pips=0.0,
        min_risk_pips=2.0,
        confirmation_window_start_hours=1,
        confirmation_window_end_hours=2,
        confirmation_mode="close_reclaim",
        body_strength_threshold=DEFAULT_BODY_STRENGTH_THRESHOLD,
        confirmation_pick="first",
        level_profile="all_levels",
        news_mode="sweep_plus_minus_30m",
        truth_model=True,
    )


def build_variants(
    *,
    start_date: str,
    end_date: str,
    profile: str = "axis_scan",
    max_variants: int | None = None,
) -> list[TruthModelConfig]:
    baseline = build_baseline_config(start_date, end_date)
    variants: list[TruthModelConfig] = [baseline]

    if profile == "axis_scan":
        for tp_r in TP_VALUES:
            if tp_r != baseline.tp_r:
                variants.append(replace(baseline, tp_r=tp_r, profile_name=profile, truth_model=False))
        for timeout_hours in TIMEOUT_VALUES:
            if timeout_hours != baseline.timeout_hours:
                variants.append(replace(baseline, timeout_hours=timeout_hours, profile_name=profile, truth_model=False))
        for sl_buffer_pips in SL_BUFFER_VALUES:
            if sl_buffer_pips != baseline.sl_buffer_pips:
                variants.append(replace(baseline, sl_buffer_pips=sl_buffer_pips, profile_name=profile, truth_model=False))
        for long_entry_buffer_pips in LONG_ENTRY_BUFFER_VALUES:
            if long_entry_buffer_pips != baseline.long_entry_buffer_pips:
                variants.append(replace(baseline, long_entry_buffer_pips=long_entry_buffer_pips, profile_name=profile, truth_model=False))
        for start_hours, end_hours in CONFIRMATION_WINDOWS:
            if (start_hours, end_hours) != (
                baseline.confirmation_window_start_hours,
                baseline.confirmation_window_end_hours,
            ):
                variants.append(
                    replace(
                        baseline,
                        confirmation_window_start_hours=start_hours,
                        confirmation_window_end_hours=end_hours,
                        profile_name=profile,
                        truth_model=False,
                    )
                )
        for confirmation_mode in CONFIRMATION_MODES:
            if confirmation_mode != baseline.confirmation_mode:
                variants.append(replace(baseline, confirmation_mode=confirmation_mode, profile_name=profile, truth_model=False))
        for confirmation_pick in CONFIRMATION_PICKS:
            if confirmation_pick != baseline.confirmation_pick:
                variants.append(replace(baseline, confirmation_pick=confirmation_pick, profile_name=profile, truth_model=False))
        for level_profile in LEVEL_PROFILES:
            if level_profile != baseline.level_profile:
                variants.append(replace(baseline, level_profile=level_profile, profile_name=profile, truth_model=False))
        for news_mode in NEWS_MODES:
            if news_mode != baseline.news_mode:
                variants.append(replace(baseline, news_mode=news_mode, profile_name=profile, truth_model=False))
    elif profile == "full_factorial":
        variants = []
        for values in product(
            TP_VALUES,
            TIMEOUT_VALUES,
            SL_BUFFER_VALUES,
            LONG_ENTRY_BUFFER_VALUES,
            CONFIRMATION_WINDOWS,
            CONFIRMATION_MODES,
            CONFIRMATION_PICKS,
            LEVEL_PROFILES,
            NEWS_MODES,
        ):
            tp_r, timeout_hours, sl_buffer_pips, long_entry_buffer_pips, window, mode, pick, level_profile, news_mode = values
            start_hours, end_hours = window
            variant = replace(
                baseline,
                profile_name=profile,
                tp_r=tp_r,
                timeout_hours=timeout_hours,
                sl_buffer_pips=sl_buffer_pips,
                long_entry_buffer_pips=long_entry_buffer_pips,
                confirmation_window_start_hours=start_hours,
                confirmation_window_end_hours=end_hours,
                confirmation_mode=mode,
                confirmation_pick=pick,
                level_profile=level_profile,
                news_mode=news_mode,
                truth_model=False,
            )
            variants.append(variant)
    else:
        raise ValueError(f"Perfil de matriz no soportado: {profile}")

    deduped: list[TruthModelConfig] = []
    seen_ids: set[str] = set()
    for variant in variants:
        candidate_id = variant.variant_id if variant.truth_model else _variant_id(variant)
        normalized = replace(variant, variant_id=candidate_id)
        if candidate_id in seen_ids:
            continue
        seen_ids.add(candidate_id)
        deduped.append(normalized)

    if max_variants is not None:
        deduped = deduped[:max_variants]
    return deduped
