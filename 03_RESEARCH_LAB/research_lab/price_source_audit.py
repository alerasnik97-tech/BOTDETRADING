from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from research_lab.config import DEFAULT_DATA_DIRS, DEFAULT_PAIR
from research_lab.data_loader import describe_available_price_data


PRICE_FILE_RE = re.compile(
    r"^(?P<pair>[A-Z]{6})_(?P<timeframe>M1|M5|M15|H1|D1|TICK|TICKS)(?:_(?P<side>BID|ASK|MID))?(?:_(?P<flavor>RAW|PREPARED))?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PriceSourceCandidate:
    path: str
    provider: str
    pair: str
    timeframe: str
    price_type: str
    side_coverage: str
    granularity_rank: int
    precision_score: int
    notes: str


def _infer_provider(path: Path, described_row: dict[str, Any] | None = None) -> str:
    manifest_source = str((described_row or {}).get("manifest_source") or "").strip().lower()
    if manifest_source:
        return manifest_source
    lowered = str(path).lower()
    if "dukascopy" in lowered:
        return "dukascopy"
    if "truefx" in lowered:
        return "truefx"
    return "unknown"


def _infer_side_coverage(path: Path, described_row: dict[str, Any] | None = None) -> tuple[str, str]:
    manifest_price_type = str((described_row or {}).get("manifest_price_type") or "").strip().lower()
    if manifest_price_type in {"bid", "ask", "mid", "bid_ask", "bid+ask"}:
        normalized = "bid_ask" if manifest_price_type in {"bid_ask", "bid+ask"} else manifest_price_type
        return normalized, manifest_price_type

    match = PRICE_FILE_RE.match(path.stem.upper())
    side = (match.group("side") if match else "") or ""
    name_upper = path.name.upper()
    sibling_bid = path.with_name(name_upper.replace("_ASK", "_BID"))
    sibling_ask = path.with_name(name_upper.replace("_BID", "_ASK"))
    if side == "ASK" and sibling_bid.exists():
        return "bid_ask", "paired_bid_ask_files"
    if side == "BID" and sibling_ask.exists():
        return "bid_ask", "paired_bid_ask_files"
    if side in {"BID", "ASK", "MID"}:
        return side.lower(), side.lower()

    if "_ASK" in name_upper and sibling_bid.exists():
        return "bid_ask", "paired_bid_ask_files"
    if "_BID" in name_upper and sibling_ask.exists():
        return "bid_ask", "paired_bid_ask_files"

    return "unknown", "no_explicit_side_metadata"


def _granularity_rank(timeframe: str, side_coverage: str) -> int:
    normalized = timeframe.upper()
    if normalized in {"TICK", "TICKS"} and side_coverage == "bid_ask":
        return 100
    if normalized == "M1" and side_coverage == "bid_ask":
        return 90
    if normalized in {"TICK", "TICKS"}:
        return 80
    if normalized == "M1":
        return 70
    if normalized == "M5":
        return 50
    if normalized == "M15":
        return 35
    if normalized == "H1":
        return 20
    return 10


def _precision_score(provider: str, timeframe: str, side_coverage: str) -> int:
    base = _granularity_rank(timeframe, side_coverage)
    provider_bonus = 0
    if provider == "dukascopy":
        provider_bonus = 5
    elif provider == "truefx":
        provider_bonus = 3
    return base + provider_bonus


def _build_candidate(path: Path, pair: str, described_row: dict[str, Any] | None = None) -> PriceSourceCandidate | None:
    match = PRICE_FILE_RE.match(path.stem.upper())
    timeframe = str((described_row or {}).get("timeframe") or "")
    if not timeframe and match:
        timeframe = str(match.group("timeframe"))
    if not timeframe:
        return None
    provider = _infer_provider(path, described_row)
    side_coverage, note = _infer_side_coverage(path, described_row)
    granularity_rank = _granularity_rank(timeframe, side_coverage)
    return PriceSourceCandidate(
        path=str(path),
        provider=provider,
        pair=pair,
        timeframe=timeframe,
        price_type=str(((described_row or {}).get("manifest_price_type") or side_coverage or "unknown")),
        side_coverage=side_coverage,
        granularity_rank=granularity_rank,
        precision_score=_precision_score(provider, timeframe, side_coverage),
        notes=note,
    )


def discover_price_sources(pair: str, data_dirs: list[Path]) -> list[PriceSourceCandidate]:
    candidates: list[PriceSourceCandidate] = []
    seen_paths: set[str] = set()
    for row in describe_available_price_data(pair, data_dirs):
        path = Path(str(row["path"]))
        candidate = _build_candidate(path, pair, row)
        if candidate is None:
            continue
        candidates.append(candidate)
        seen_paths.add(str(path.resolve()))
    for root in data_dirs:
        if not root.exists():
            continue
        for path in root.rglob(f"{pair.upper()}*.csv"):
            resolved = str(path.resolve())
            if resolved in seen_paths:
                continue
            candidate = _build_candidate(path, pair)
            if candidate is None:
                continue
            candidates.append(candidate)
            seen_paths.add(resolved)
    return sorted(candidates, key=lambda item: (-item.precision_score, item.path))


def build_price_source_recommendation(pair: str, data_dirs: list[Path]) -> dict[str, Any]:
    discovered = discover_price_sources(pair, data_dirs)
    current_best = asdict(discovered[0]) if discovered else None
    has_bid_ask = any(item.side_coverage == "bid_ask" for item in discovered)
    has_m1 = any(item.timeframe.upper() == "M1" for item in discovered)
    has_tick = any(item.timeframe.upper() in {"TICK", "TICKS"} for item in discovered)

    target_primary = {
        "provider": "dukascopy",
        "recommended_asset": "tick bid+ask o M1 bid+ask",
        "why": "es la continuidad mas natural respecto a la base actual y permite modelar ejecucion sobre ambos lados del mercado en vez de sintetizar ASK desde BID",
    }
    target_secondary = {
        "provider": "truefx",
        "recommended_role": "validacion externa de timestamps, estructura bid/offer y distribucion horaria de spreads",
        "why": "sirve como contraste independiente para detectar sesgos en Dukascopy o en el pipeline local; no conviene usarlo como fuente principal sin un pipeline de normalizacion dedicado",
    }

    current_limitations: list[str] = []
    if not discovered:
        current_limitations.append("no_hay_fuentes_de_precios_descubiertas_en_los_data_dirs_actuales")
    if not has_bid_ask:
        current_limitations.append("no_hay_BID_ASK_historico_local; el ASK sigue siendo sintetico")
    if not has_m1:
        current_limitations.append("no_hay_granularidad_M1_local; la base actual opera sobre M5 preparado")
    if not has_tick:
        current_limitations.append("no_hay_tick_data_local; la politica intrabar sigue dependiendo de OHLC")

    return {
        "pair": pair,
        "data_dirs": [str(path) for path in data_dirs],
        "discovered_sources": [asdict(item) for item in discovered],
        "current_best_local_source": current_best,
        "recommended_primary_upgrade": target_primary,
        "recommended_secondary_validation": target_secondary,
        "current_limitations": current_limitations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audita fuentes locales de precios y recomienda el siguiente escalon de precision.")
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--data-dirs", nargs="*", default=[str(path) for path in DEFAULT_DATA_DIRS])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = build_price_source_recommendation(args.pair.upper().strip(), [Path(value) for value in args.data_dirs])
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
