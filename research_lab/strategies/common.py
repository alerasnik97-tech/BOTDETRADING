from __future__ import annotations

import math
from itertools import product
from typing import Iterable

import numpy as np

from research_lab.config import SESSION_VARIANTS


def finite(value: float) -> bool:
    return value == value and math.isfinite(value)


def day_range_ok(frame, i: int, min_atr: float, max_atr: float, use_h1_scale: bool = False) -> bool:
    column = "day_range_h1_atr" if use_h1_scale else "day_range_m15_atr"
    value = float(frame[column].iat[i])
    return finite(value) and min_atr <= value <= max_atr


def candle_not_extended(frame, i: int, max_range_atr: float) -> bool:
    value = float(frame["range_atr"].iat[i])
    return finite(value) and value <= max_range_atr


def h1_filter_passes(frame, i: int, use_h1_context: bool, direction: str, ema_filter: int | None = None, adx_min: float | None = None) -> bool:
    if not use_h1_context:
        return True
    if direction == "long":
        if not (float(frame["h1_ema50"].iat[i]) > float(frame["h1_ema200"].iat[i]) and float(frame["h1_ema200_slope_5"].iat[i]) > 0):
            return False
        if ema_filter == 100 and not (float(frame["close"].iat[i]) > float(frame["h1_ema100"].iat[i])):
            return False
        if ema_filter == 200 and not (float(frame["close"].iat[i]) > float(frame["h1_ema200"].iat[i])):
            return False
    else:
        if not (float(frame["h1_ema50"].iat[i]) < float(frame["h1_ema200"].iat[i]) and float(frame["h1_ema200_slope_5"].iat[i]) < 0):
            return False
        if ema_filter == 100 and not (float(frame["close"].iat[i]) < float(frame["h1_ema100"].iat[i])):
            return False
        if ema_filter == 200 and not (float(frame["close"].iat[i]) < float(frame["h1_ema200"].iat[i])):
            return False
    if adx_min is not None and float(frame["h1_adx14"].iat[i]) < float(adx_min):
        return False
    return True


def session_window(name: str) -> tuple[str, str]:
    return SESSION_VARIANTS[name]


def add_general_params(param_space: dict[str, list]) -> dict[str, list]:
    merged = dict(param_space)
    merged["session_name"] = list(SESSION_VARIANTS.keys())
    merged["use_h1_context"] = [False, True]
    merged["break_even_at_r"] = [None, 1.0, 1.2]
    return merged


def cartesian_product(param_space: dict[str, list]) -> list[dict]:
    keys = list(param_space.keys())
    return [dict(zip(keys, values)) for values in product(*(param_space[key] for key in keys))]


def stratified_sample_combinations(param_space: dict[str, list], max_samples: int, seed: int) -> list[dict]:
    all_combos = cartesian_product(param_space)
    if len(all_combos) <= max_samples:
        return all_combos

    rng = np.random.default_rng(seed)
    chosen: list[dict] = []
    seen_keys: set[tuple] = set()

    for key, values in param_space.items():
        for value in values:
            candidates = [combo for combo in all_combos if combo[key] == value]
            if not candidates:
                continue
            picked = candidates[int(rng.integers(0, len(candidates)))]
            signature = tuple((k, picked[k]) for k in sorted(picked))
            if signature not in seen_keys:
                chosen.append(picked)
                seen_keys.add(signature)
            if len(chosen) >= max_samples:
                return chosen

    remaining = [combo for combo in all_combos if tuple((k, combo[k]) for k in sorted(combo)) not in seen_keys]
    if remaining:
        rng.shuffle(remaining)
        for combo in remaining:
            chosen.append(combo)
            if len(chosen) >= max_samples:
                break
    return chosen[:max_samples]


def bool_off_on() -> list[bool]:
    return [False, True]
