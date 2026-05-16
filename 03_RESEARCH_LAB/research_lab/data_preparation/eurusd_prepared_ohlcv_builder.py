from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised only on broken local envs.
    pq = None  # type: ignore[assignment]
    _PYARROW_IMPORT_ERROR = exc
else:
    _PYARROW_IMPORT_ERROR = None


PAIR = "EURUSD"
BUILDER_VERSION = "eurusd_prepared_ohlcv_builder_v1_20260516"
RAW_MONTHLY_RE = re.compile(r"^EURUSD_ticks_(?P<year>\d{4})_(?P<month>\d{2})\.parquet$")
DEFAULT_RAW_DIR = Path("05_MARKET_DATA_VAULT/BOT_MARKET_DATA/tick/EURUSD")
DEFAULT_OUTPUT_DIR = Path("05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared")
DEFAULT_GOVERNANCE_MANIFEST = Path("06_GOVERNANCE_AND_COMPLIANCE/lab_readiness/EURUSD_PREPARED_OHLCV_MANIFEST.csv")
LOCAL_MANIFEST_NAME = "MANIFEST_LOCAL_DO_NOT_COMMIT.json"
LOADER_MANIFEST_NAME = "prepared_data_manifest.json"
RAW_COLUMNS = ("timestamp_utc", "bid", "ask", "bid_volume", "ask_volume", "source", "symbol")
REQUIRED_RAW_COLUMNS = ("timestamp_utc", "bid", "ask")
OUTPUT_COLUMNS = ("open", "high", "low", "close", "volume")
TIMEFRAME_RULES = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
}


@dataclass(frozen=True)
class SourceFile:
    path: Path
    year: int
    month: int
    size_bytes: int


@dataclass(frozen=True)
class RawDiscovery:
    raw_dir: str
    files_found_total: int
    monthly_files_found: int
    included_files: list[str]
    excluded_2025_2026_files: list[str]
    skipped_non_monthly_files_count: int
    skipped_non_monthly_files_sample: list[str]
    min_source_period: str | None
    max_source_period: str | None
    included_min_period: str | None
    included_max_period: str | None
    total_size_bytes: int
    included_size_bytes: int


@dataclass
class BuildStats:
    source_files_used: int = 0
    source_files_excluded_2025_2026: int = 0
    source_files_skipped_non_monthly: int = 0
    raw_rows_read: int = 0
    raw_rows_kept: int = 0
    exact_duplicate_ticks_removed: int = 0
    duplicate_timestamps_detected_after_exact_dedup: int = 0
    rows_filtered_by_max_date: int = 0
    min_raw_timestamp_utc: str | None = None
    max_raw_timestamp_utc: str | None = None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_max_exclusive(max_date: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(max_date)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    if timestamp.time() == datetime.min.time():
        timestamp = timestamp + pd.Timedelta(days=1)
    return timestamp


def _period_label(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _file_period(path: Path) -> tuple[int, int] | None:
    match = RAW_MONTHLY_RE.match(path.name)
    if match is None:
        return None
    return int(match.group("year")), int(match.group("month"))


def discover_raw_files(raw_dir: Path, *, max_exclusive_utc: pd.Timestamp) -> tuple[RawDiscovery, list[SourceFile]]:
    all_files = sorted([path for path in raw_dir.rglob("*") if path.is_file()])
    source_files: list[SourceFile] = []
    excluded: list[str] = []
    skipped: list[str] = []
    monthly_periods: list[tuple[int, int]] = []

    for path in all_files:
        period = _file_period(path)
        if period is None:
            skipped.append(str(path))
            continue
        year, month = period
        monthly_periods.append(period)
        size_bytes = int(path.stat().st_size)
        period_start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        if period_start >= max_exclusive_utc:
            excluded.append(str(path))
            continue
        if year >= 2025:
            excluded.append(str(path))
            continue
        source_files.append(SourceFile(path=path, year=year, month=month, size_bytes=size_bytes))

    periods_sorted = sorted(monthly_periods)
    included_periods = sorted((item.year, item.month) for item in source_files)
    discovery = RawDiscovery(
        raw_dir=str(raw_dir),
        files_found_total=len(all_files),
        monthly_files_found=len(monthly_periods),
        included_files=[str(item.path) for item in source_files],
        excluded_2025_2026_files=excluded,
        skipped_non_monthly_files_count=len(skipped),
        skipped_non_monthly_files_sample=skipped[:25],
        min_source_period=_period_label(*periods_sorted[0]) if periods_sorted else None,
        max_source_period=_period_label(*periods_sorted[-1]) if periods_sorted else None,
        included_min_period=_period_label(*included_periods[0]) if included_periods else None,
        included_max_period=_period_label(*included_periods[-1]) if included_periods else None,
        total_size_bytes=sum(int(path.stat().st_size) for path in all_files),
        included_size_bytes=sum(item.size_bytes for item in source_files),
    )
    return discovery, source_files


def read_parquet_schema(path: Path) -> dict[str, str]:
    if pq is None:
        raise RuntimeError(f"pyarrow is required to inspect parquet schema: {_PYARROW_IMPORT_ERROR}")
    schema = pq.read_schema(path)
    return {name: str(schema.field(name).type) for name in schema.names}


def detect_schema(source_files: list[SourceFile]) -> dict[str, Any]:
    if not source_files:
        raise ValueError("RAW_EURUSD_SCHEMA_BLOCKER: no included monthly EURUSD parquet files were found.")
    schema = read_parquet_schema(source_files[0].path)
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in schema]
    if missing:
        raise ValueError(f"RAW_EURUSD_SCHEMA_BLOCKER: missing raw columns {missing} in {source_files[0].path}")
    return {
        "sample_file": str(source_files[0].path),
        "schema": schema,
        "required_columns_present": True,
        "price_construction": "mid=(bid+ask)/2",
        "volume_construction": "observed_tick_count_per_bar",
        "timestamp_source": "timestamp_utc",
        "timestamp_timezone": schema.get("timestamp_utc", "unknown"),
    }


def _read_source_frame(path: Path) -> pd.DataFrame:
    schema = read_parquet_schema(path)
    columns = [column for column in RAW_COLUMNS if column in schema]
    frame = pd.read_parquet(path, columns=columns)
    missing = [column for column in REQUIRED_RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"RAW_EURUSD_SCHEMA_BLOCKER: missing columns {missing} in {path}")
    return frame


def _clean_ticks(frame: pd.DataFrame, source: SourceFile, max_exclusive_utc: pd.Timestamp, stats: BuildStats) -> pd.DataFrame:
    stats.raw_rows_read += int(len(frame))
    required_nulls = frame[list(REQUIRED_RAW_COLUMNS)].isna().any(axis=1)
    if bool(required_nulls.any()):
        raise ValueError(f"RAW_EURUSD_SCHEMA_BLOCKER: null timestamp/bid/ask rows in {source.path}: {int(required_nulls.sum())}")

    if "symbol" in frame.columns:
        symbols = set(str(value) for value in frame["symbol"].dropna().unique())
        if symbols and symbols != {PAIR}:
            raise ValueError(f"RAW_EURUSD_SCHEMA_BLOCKER: unexpected symbols in {source.path}: {sorted(symbols)}")

    bid_gt_ask = frame["bid"].astype(float) > frame["ask"].astype(float)
    if bool(bid_gt_ask.any()):
        raise ValueError(f"RAW_EURUSD_SCHEMA_BLOCKER: bid greater than ask in {source.path}: {int(bid_gt_ask.sum())}")

    frame = frame.copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    before_max = len(frame)
    frame = frame.loc[frame["timestamp_utc"] < max_exclusive_utc].copy()
    stats.rows_filtered_by_max_date += int(before_max - len(frame))
    if frame.empty:
        return frame

    exact_subset = [column for column in ("timestamp_utc", "bid", "ask", "bid_volume", "ask_volume") if column in frame.columns]
    exact_duplicate_mask = frame.duplicated(subset=exact_subset, keep="first")
    stats.exact_duplicate_ticks_removed += int(exact_duplicate_mask.sum())
    if bool(exact_duplicate_mask.any()):
        frame = frame.loc[~exact_duplicate_mask].copy()

    duplicate_timestamps = frame["timestamp_utc"].duplicated(keep=False)
    stats.duplicate_timestamps_detected_after_exact_dedup += int(duplicate_timestamps.sum())
    frame["mid"] = (frame["bid"].astype(float) + frame["ask"].astype(float)) / 2.0
    frame = frame[["timestamp_utc", "mid"]].sort_values("timestamp_utc")
    stats.raw_rows_kept += int(len(frame))

    min_ts = frame["timestamp_utc"].min()
    max_ts = frame["timestamp_utc"].max()
    if stats.min_raw_timestamp_utc is None or str(min_ts) < stats.min_raw_timestamp_utc:
        stats.min_raw_timestamp_utc = str(min_ts)
    if stats.max_raw_timestamp_utc is None or str(max_ts) > stats.max_raw_timestamp_utc:
        stats.max_raw_timestamp_utc = str(max_ts)
    return frame


def _resample_ticks_to_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    index = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="raise")
    series = pd.Series(frame["mid"].to_numpy(dtype=float), index=index, name="mid").sort_index()
    ohlc = series.resample(rule, label="right", closed="right").ohlc()
    volume = series.resample(rule, label="right", closed="right").count().rename("volume")
    bars = pd.concat([ohlc, volume], axis=1)
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars.index.name = "timestamp"
    return bars.loc[:, OUTPUT_COLUMNS]


def _merge_bars(frames: list[pd.DataFrame], max_exclusive_utc: pd.Timestamp) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    merged = pd.concat(frames).sort_index()
    merged = merged.loc[merged.index < max_exclusive_utc].copy()
    if merged.empty:
        return merged
    merged = (
        merged.groupby(level=0, sort=True)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .loc[:, OUTPUT_COLUMNS]
    )
    merged.index.name = "timestamp"
    return merged


def _gap_summary(index: pd.DatetimeIndex, rule: str) -> str:
    if len(index) < 2:
        return "not_enough_rows"
    expected = pd.Timedelta(rule)
    deltas = pd.Series(index[1:] - index[:-1])
    gaps = deltas.loc[deltas > expected]
    if gaps.empty:
        return "no_gaps_gt_expected_delta"
    return (
        f"gaps_gt_expected_delta={len(gaps)};"
        f"max_gap={str(gaps.max())};"
        f"first_gap={str(index[int(gaps.index[0])])}"
    )


def _validate_output_frame(frame: pd.DataFrame, timeframe: str, max_exclusive_utc: pd.Timestamp) -> None:
    if frame.empty:
        raise ValueError(f"BUILDER_FAILURE: {timeframe} output frame is empty.")
    if frame.index.tz is None:
        raise ValueError(f"BUILDER_FAILURE: {timeframe} output index is timezone-naive.")
    if bool((frame.index >= max_exclusive_utc).any()):
        raise ValueError(f"EURUSD_PREPARED_OHLCV_BLOCKED_2025_2026_LEAKAGE: {timeframe} contains >= {max_exclusive_utc}")
    if frame.index.duplicated().any():
        raise ValueError(f"BUILDER_FAILURE: {timeframe} output contains duplicate timestamps.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"BUILDER_FAILURE: {timeframe} output index is not monotonic.")
    invalid = (frame["high"] < frame[["open", "close", "low"]].max(axis=1)) | (
        frame["low"] > frame[["open", "close", "high"]].min(axis=1)
    )
    if bool(invalid.any()):
        raise ValueError(f"BUILDER_FAILURE: {timeframe} contains invalid OHLC rows: {int(invalid.sum())}")


def _write_csv_atomic(frame: pd.DataFrame, path: Path, *, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing prepared data without --allow-overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    frame.to_csv(tmp_path, float_format="%.10f")
    tmp_path.replace(path)


def build_prepared_ohlcv(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_date: str = "2024-12-31",
    dry_run: bool = False,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    max_exclusive_utc = _normalize_max_exclusive(max_date)
    discovery, source_files = discover_raw_files(raw_dir, max_exclusive_utc=max_exclusive_utc)
    schema_info = detect_schema(source_files)
    stats = BuildStats(
        source_files_used=len(source_files),
        source_files_excluded_2025_2026=len(discovery.excluded_2025_2026_files),
        source_files_skipped_non_monthly=discovery.skipped_non_monthly_files_count,
    )
    base_summary: dict[str, Any] = {
        "status": "DRY_RUN_OK" if dry_run else "BUILD_PENDING",
        "builder_version": BUILDER_VERSION,
        "build_timestamp_utc": utc_now_iso(),
        "pair": PAIR,
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "max_date_argument": max_date,
        "max_timestamp_exclusive_utc": str(max_exclusive_utc),
        "raw_discovery": asdict(discovery),
        "raw_schema": schema_info,
        "timeframes": {},
        "stats": asdict(stats),
        "safety": {
            "train_only": True,
            "excluded_2025_2026_by_filename": True,
            "excluded_rows_at_or_after_max_timestamp": True,
            "price_synthesized": False,
            "gaps_forward_filled": False,
            "empty_bars_fabricated": False,
            "volume_source": "observed tick_count per bar",
        },
    }
    if dry_run:
        base_summary["target_files"] = [str(output_dir / f"{PAIR}_{timeframe}.csv") for timeframe in TIMEFRAME_RULES]
        return base_summary

    per_timeframe_frames: dict[str, list[pd.DataFrame]] = {timeframe: [] for timeframe in TIMEFRAME_RULES}
    for source in source_files:
        raw_frame = _read_source_frame(source.path)
        ticks = _clean_ticks(raw_frame, source, max_exclusive_utc, stats)
        if ticks.empty:
            continue
        for timeframe, rule in TIMEFRAME_RULES.items():
            per_timeframe_frames[timeframe].append(_resample_ticks_to_ohlcv(ticks, rule))

    output_manifest: dict[str, Any] = {}
    for timeframe, rule in TIMEFRAME_RULES.items():
        frame = _merge_bars(per_timeframe_frames[timeframe], max_exclusive_utc)
        _validate_output_frame(frame, timeframe, max_exclusive_utc)
        output_path = output_dir / f"{PAIR}_{timeframe}.csv"
        _write_csv_atomic(frame, output_path, allow_overwrite=allow_overwrite)
        output_manifest[timeframe] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "row_count": int(len(frame)),
            "min_timestamp_utc": str(frame.index.min()),
            "max_timestamp_utc": str(frame.index.max()),
            "schema": list(OUTPUT_COLUMNS),
            "gaps_summary": _gap_summary(frame.index, rule),
            "duplicate_bars_collapsed": int(sum(len(item) for item in per_timeframe_frames[timeframe]) - len(frame)),
        }

    stats_payload = asdict(stats)
    summary = {
        **base_summary,
        "status": "BUILT_OK",
        "build_timestamp_utc": utc_now_iso(),
        "timeframes": output_manifest,
        "stats": stats_payload,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for manifest_name in (LOCAL_MANIFEST_NAME, LOADER_MANIFEST_NAME):
        manifest_path = output_dir / manifest_name
        if manifest_path.exists() and not allow_overwrite:
            raise FileExistsError(f"Refusing to overwrite existing manifest without --allow-overwrite: {manifest_path}")
        tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
        tmp_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(manifest_path)
    return summary


def write_governance_manifest(summary: dict[str, Any], manifest_path: Path = DEFAULT_GOVERNANCE_MANIFEST) -> None:
    raw_discovery = summary["raw_discovery"]
    stats = summary["stats"]
    rows: list[dict[str, Any]] = []
    for timeframe, payload in summary["timeframes"].items():
        rows.append(
            {
                "timeframe": timeframe,
                "output_path": payload["path"],
                "sha256": payload["sha256"],
                "row_count": payload["row_count"],
                "min_timestamp_utc": payload["min_timestamp_utc"],
                "max_timestamp_utc": payload["max_timestamp_utc"],
                "source_files_used": stats["source_files_used"],
                "source_files_excluded_2025_2026": stats["source_files_excluded_2025_2026"],
                "duplicate_rows_removed": stats["exact_duplicate_ticks_removed"],
                "gaps_summary": payload["gaps_summary"],
                "schema": "|".join(payload["schema"]),
                "git_tracked": "NO",
                "notes": "local_train_only_ohlcv_from_raw_tick_mid_price_volume_tick_count",
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    pd.DataFrame(rows).to_csv(tmp_path, index=False)
    tmp_path.replace(manifest_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train-only EURUSD prepared OHLCV from local raw tick parquet.")
    parser.add_argument("--raw-dir", default=str(DEFAULT_RAW_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-date", default="2024-12-31")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--write-governance-manifest", action="store_true")
    parser.add_argument("--governance-manifest", default=str(DEFAULT_GOVERNANCE_MANIFEST))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    summary = build_prepared_ohlcv(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        max_date=args.max_date,
        dry_run=bool(args.dry_run),
        allow_overwrite=bool(args.allow_overwrite),
    )
    if args.write_governance_manifest and not args.dry_run:
        write_governance_manifest(summary, Path(args.governance_manifest))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
