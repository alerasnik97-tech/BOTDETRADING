from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PIP_SIZE = 0.0001
DEFAULT_BODY_STRENGTH_THRESHOLD = 0.50

LEVEL_PROFILE_MAP = {
    "all_levels": ("pd", "asia", "london"),
    "pd_only": ("pd",),
    "asia_only": ("asia",),
    "london_only": ("london",),
}

NEWS_MODE_LABELS = {
    "none": "Sin filtro de noticias",
    "sweep_plus_minus_30m": "Bloqueo por noticia +/-30m alrededor del sweep",
    "sweep_plus_minus_60m": "Bloqueo por noticia +/-60m alrededor del sweep",
    "post_news_cooldown_60m": "Cooldown post noticia de 60m desde el evento",
}

CONFIRMATION_MODE_LABELS = {
    "close_reclaim": "Close M5 del lado correcto del nivel",
    "close_reclaim_body_strength": "Close M5 del lado correcto + body strength minima",
}

CONFIRMATION_PICK_LABELS = {
    "first": "Primera confirmacion elegible",
    "best": "Mejor confirmacion por desplazamiento de cierre",
}


@dataclass(frozen=True)
class HarnessPaths:
    workspace_root: Path
    core_root: Path
    price_dirs: tuple[Path, ...]
    news_file: Path
    output_root: Path
    cache_root: Path


@dataclass(frozen=True)
class TruthModelConfig:
    variant_id: str = "baseline_truth_model"
    profile_name: str = "baseline"
    start_date: str = "2020-01-01"
    end_date: str = "2025-12-31"
    tp_r: float = 1.5
    timeout_hours: int = 4
    sl_buffer_pips: float = 1.0
    long_entry_buffer_pips: float = 0.3
    short_entry_buffer_pips: float = 0.0
    min_risk_pips: float = 2.0
    confirmation_window_start_hours: int = 1
    confirmation_window_end_hours: int = 2
    confirmation_mode: str = "close_reclaim"
    body_strength_threshold: float = DEFAULT_BODY_STRENGTH_THRESHOLD
    confirmation_pick: str = "first"
    level_profile: str = "all_levels"
    news_mode: str = "sweep_plus_minus_30m"
    truth_model: bool = False

    @property
    def allowed_level_groups(self) -> tuple[str, ...]:
        return LEVEL_PROFILE_MAP[self.level_profile]

    @property
    def confirmation_window_label(self) -> str:
        return f"+{self.confirmation_window_start_hours}h_+{self.confirmation_window_end_hours}h"


def default_paths(workspace_root: Path | None = None) -> HarnessPaths:
    root = (workspace_root or Path(__file__).resolve().parent.parent).resolve()
    core_root = root / "BOT DE TRADING ultimo"
    return HarnessPaths(
        workspace_root=root,
        core_root=core_root,
        price_dirs=(
            core_root / "data_free_2020" / "prepared",
            core_root / "data_candidates_2022_2025" / "prepared",
        ),
        news_file=core_root / "data" / "news_eurusd_am_fortress_v3.csv",
        output_root=root / "external_scbi_research_harness" / "outputs",
        cache_root=root / "external_scbi_research_harness" / "cache",
    )
