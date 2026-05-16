from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PIP_SIZE = 0.0001
DEFAULT_BODY_STRENGTH_THRESHOLD = 0.50
DEFAULT_START_DATE = "2020-01-01"
DEFAULT_END_DATE = "2025-12-31"

LEVEL_PROFILE_MAP = {
    "all_levels": ("pd", "asia", "london"),
    "pd_only": ("pd",),
    "asia_only": ("asia",),
    "london_only": ("london",),
}

LEVEL_PROFILE_LABELS = {
    "all_levels": "PD + Asia + London",
    "pd_only": "Solo PDH/PDL",
    "asia_only": "Solo Asia",
    "london_only": "Solo London",
}

NEWS_MODE_LABELS = {
    "none": "Sin filtro de noticias",
    "sweep_plus_minus_15m": "Bloqueo +/-15m alrededor del sweep",
    "sweep_plus_minus_30m": "Bloqueo +/-30m alrededor del sweep",
    "sweep_plus_minus_60m": "Bloqueo +/-60m alrededor del sweep",
}

CONFIRMATION_MODE_LABELS = {
    "close_reclaim": "Reclaim simple por close M5",
    "close_reclaim_body_strength": "Reclaim + cuerpo minimo M5 (experimental)",
}

CONFIRMATION_PICK_LABELS = {
    "first": "Primera confirmacion elegible",
    "best": "Mejor confirmacion dentro de la ventana",
}


@dataclass(frozen=True)
class LabPaths:
    project_root: Path
    lab_root: Path
    output_root: Path
    tests_root: Path
    price_dirs: tuple[Path, ...]
    news_file: Path
    canonical_zip: Path


@dataclass(frozen=True)
class CandidateConfig:
    variant_id: str = "baseline_truth_model"
    profile_name: str = "baseline"
    start_date: str = DEFAULT_START_DATE
    end_date: str = DEFAULT_END_DATE
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
    research_status: str = "RESEARCH_ONLY"
    promotion_status: str = "NO_PRODUCTION"
    truth_model: bool = False
    experimental_variant: bool = False

    @property
    def allowed_level_groups(self) -> tuple[str, ...]:
        return LEVEL_PROFILE_MAP[self.level_profile]

    @property
    def confirmation_window_label(self) -> str:
        return f"+{self.confirmation_window_start_hours}h_+{self.confirmation_window_end_hours}h"


def default_paths(project_root: Path | None = None) -> LabPaths:
    root = (project_root or Path(__file__).resolve().parent.parent).resolve()
    lab_root = root / "institutional_research_candidate_lab"
    return LabPaths(
        project_root=root,
        lab_root=lab_root,
        output_root=lab_root / "outputs",
        tests_root=lab_root / "tests",
        price_dirs=(
            root / "data_free_2020" / "prepared",
            root / "data_candidates_2022_2025" / "prepared",
        ),
        news_file=root / "data" / "news_eurusd_am_fortress_v3.csv",
        canonical_zip=root / "000_PARA_CHATGPT.zip",
    )
