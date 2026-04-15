from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Iterable

import pandas as pd

import dukascopy_python as dukascopy

from research_lab.config import DEFAULT_HIGH_PRECISION_RAW_DIR
from research_lab.data_loader import OHLCV_COLUMNS, validate_price_frame


PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
)


@dataclass(frozen=True)
class DownloadChunk:
    start: date
    end_inclusive: date


@dataclass(frozen=True)
class DownloadChunkResult:
    side: str
    start: str
    end_inclusive: str
    status: str
    attempts: int
    rows: int
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None
    error: str | None = None


def _clear_proxy_env() -> None:
    for key in PROXY_ENV_VARS:
        os.environ.pop(key, None)


def _dukascopy_instrument(pair: str) -> str:
    normalized = pair.upper().strip()
    if len(normalized) != 6:
        raise ValueError(f"Par no soportado para instrumentacion automatica: {pair}")
    return f"{normalized[:3]}/{normalized[3:]}"


def _next_month_start(current: date) -> date:
    if current.month == 12:
        return date(current.year + 1, 1, 1)
    return date(current.year, current.month + 1, 1)


def _month_chunks(start: date, end_inclusive: date) -> list[DownloadChunk]:
    chunks: list[DownloadChunk] = []
    cursor = start
    while cursor <= end_inclusive:
        next_month = _next_month_start(cursor)
        chunk_end = min(end_inclusive, next_month - timedelta(days=1))
        chunks.append(DownloadChunk(start=cursor, end_inclusive=chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


def _chunk_window(chunk: DownloadChunk) -> tuple[datetime, datetime]:
    start_dt = datetime.combine(chunk.start, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(
        chunk.end_inclusive,
        datetime.max.time().replace(second=0, microsecond=0),
        tzinfo=timezone.utc,
    )
    return start_dt, end_dt


def _fetch_chunk(instrument: str, side: str, chunk: DownloadChunk) -> pd.DataFrame:
    start_dt, end_dt = _chunk_window(chunk)
    end_exclusive = end_dt + timedelta(minutes=1)
    offer_side = dukascopy.OFFER_SIDE_BID if side == "BID" else dukascopy.OFFER_SIDE_ASK
    frame = dukascopy.fetch(
        instrument,
        dukascopy.INTERVAL_MIN_1,
        offer_side,
        start_dt,
        end_exclusive,
    )
    frame = frame.copy()
    frame = frame[list(OHLCV_COLUMNS)].astype(float)
    if frame.index.tz is None:
        raise ValueError("La descarga Dukascopy devolvio un indice sin timezone explicita.")
    frame = frame.loc[(frame.index >= start_dt) & (frame.index <= end_dt)].copy()
    validate_price_frame(frame)
    return frame


def _concat_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat(list(frames)).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    validate_price_frame(merged)
    return merged


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_manifest(
    *,
    pair: str,
    instrument: str,
    start_date: date,
    end_date: date,
    chunks: list[DownloadChunk],
    chunk_results: list[DownloadChunkResult],
    generated_files: dict[str, str] | None = None,
    rows: dict[str, int] | None = None,
    first_timestamp_utc: str | None = None,
    last_timestamp_utc: str | None = None,
    alignment: dict[str, object] | None = None,
) -> dict[str, object]:
    successful = [asdict(item) for item in chunk_results if item.status == "success"]
    failed = [asdict(item) for item in chunk_results if item.status != "success"]
    periods = {
        (chunk.start.isoformat(), chunk.end_inclusive.isoformat())
        for chunk in chunks
    }
    period_state: dict[tuple[str, str], dict[str, str]] = {period: {} for period in periods}
    for item in chunk_results:
        period = (item.start, item.end_inclusive)
        period_state.setdefault(period, {})
        period_state[period][item.side] = item.status
    fully_downloaded_periods = [
        {"start": start, "end_inclusive": end_inclusive}
        for start, end_inclusive in sorted(periods)
        if period_state.get((start, end_inclusive), {}).get("BID") == "success"
        and period_state.get((start, end_inclusive), {}).get("ASK") == "success"
    ]
    failed_periods = [
        {"start": start, "end_inclusive": end_inclusive}
        for start, end_inclusive in sorted(periods)
        if not (
            period_state.get((start, end_inclusive), {}).get("BID") == "success"
            and period_state.get((start, end_inclusive), {}).get("ASK") == "success"
        )
    ]
    return {
        "provider": "dukascopy",
        "instrument": instrument,
        "pair": pair.upper(),
        "granularity": "M1",
        "sides": ["BID", "ASK"],
        "requested_start_utc": f"{start_date.isoformat()}T00:00:00+00:00",
        "requested_end_utc_inclusive": f"{end_date.isoformat()}T23:59:00+00:00",
        "expected_months": [
            {
                "start": chunk.start.isoformat(),
                "end_inclusive": chunk.end_inclusive.isoformat(),
            }
            for chunk in chunks
        ],
        "successful_chunks": successful,
        "failed_chunks": failed,
        "downloaded_months": fully_downloaded_periods,
        "failed_months": failed_periods,
        "months_downloaded": len(fully_downloaded_periods),
        "months_failed": len(failed_periods),
        "generated_files": generated_files or {},
        "rows": rows or {},
        "first_timestamp_utc": first_timestamp_utc,
        "last_timestamp_utc": last_timestamp_utc,
        "bid_ask_alignment": alignment or {},
        "notes": [
            "descarga automatica via dukascopy_python usando instrument=EUR/USD para EURUSD",
            "las variables de proxy del entorno se limpian dentro del script para evitar el proxy roto local",
            "los CSV quedan crudos en UTC con timezone explicita y listos para high_precision_import",
        ],
    }


def download_m1_bid_ask(
    *,
    pair: str,
    start_date: date,
    end_date: date,
    output_dir: Path,
    chunk_retries: int = 3,
    retry_sleep_seconds: float = 1.5,
) -> dict[str, object]:
    if end_date < start_date:
        raise ValueError("La fecha final no puede ser menor que la inicial.")
    if chunk_retries < 1:
        raise ValueError("chunk_retries debe ser >= 1.")

    _clear_proxy_env()
    instrument = _dukascopy_instrument(pair)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = _month_chunks(start_date, end_date)
    requested_start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    requested_end_dt = datetime.combine(
        end_date,
        datetime.max.time().replace(second=0, microsecond=0),
        tzinfo=timezone.utc,
    )
    manifest_path = output_dir / f"{pair.upper()}_M1_download_manifest.json"

    payload: dict[str, pd.DataFrame] = {}
    chunk_results: list[DownloadChunkResult] = []

    for side in ("BID", "ASK"):
        side_frames: list[pd.DataFrame] = []
        for chunk in chunks:
            success = False
            last_error: str | None = None
            for attempt in range(1, chunk_retries + 1):
                try:
                    frame = _fetch_chunk(instrument, side, chunk)
                    side_frames.append(frame)
                    chunk_results.append(
                        DownloadChunkResult(
                            side=side,
                            start=chunk.start.isoformat(),
                            end_inclusive=chunk.end_inclusive.isoformat(),
                            status="success",
                            attempts=attempt,
                            rows=int(len(frame)),
                            first_timestamp_utc=frame.index.min().isoformat(),
                            last_timestamp_utc=frame.index.max().isoformat(),
                        )
                    )
                    success = True
                    break
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < chunk_retries:
                        sleep(retry_sleep_seconds)
            if not success:
                chunk_results.append(
                    DownloadChunkResult(
                        side=side,
                        start=chunk.start.isoformat(),
                        end_inclusive=chunk.end_inclusive.isoformat(),
                        status="failed",
                        attempts=chunk_retries,
                        rows=0,
                        first_timestamp_utc=None,
                        last_timestamp_utc=None,
                        error=last_error,
                    )
                )
            _write_manifest(
                manifest_path,
                _build_manifest(
                    pair=pair,
                    instrument=instrument,
                    start_date=start_date,
                    end_date=end_date,
                    chunks=chunks,
                    chunk_results=chunk_results,
                ),
            )
        if not side_frames:
            raise RuntimeError(f"No se pudo descargar ningun chunk para {pair} {side}.")
        merged = _concat_frames(side_frames)
        merged = merged.loc[(merged.index >= requested_start_dt) & (merged.index <= requested_end_dt)].copy()
        validate_price_frame(merged)
        payload[side] = merged

    common_index = payload["BID"].index.intersection(payload["ASK"].index)
    if common_index.empty:
        raise RuntimeError("BID y ASK quedaron sin timestamps comunes despues de descargar.")
    dropped_bid = int(len(payload["BID"]) - len(common_index))
    dropped_ask = int(len(payload["ASK"]) - len(common_index))
    payload["BID"] = payload["BID"].loc[common_index].copy()
    payload["ASK"] = payload["ASK"].loc[common_index].copy()
    validate_price_frame(payload["BID"])
    validate_price_frame(payload["ASK"])

    generated_files = {
        "bid": str(output_dir / f"{pair.upper()}_M1_BID.csv"),
        "ask": str(output_dir / f"{pair.upper()}_M1_ASK.csv"),
    }
    payload["BID"].to_csv(output_dir / f"{pair.upper()}_M1_BID.csv", index_label="timestamp")
    payload["ASK"].to_csv(output_dir / f"{pair.upper()}_M1_ASK.csv", index_label="timestamp")

    manifest = _build_manifest(
        pair=pair,
        instrument=instrument,
        start_date=start_date,
        end_date=end_date,
        chunks=chunks,
        chunk_results=chunk_results,
        generated_files=generated_files,
        rows={side.lower(): int(len(frame)) for side, frame in payload.items()},
        first_timestamp_utc=min(frame.index.min() for frame in payload.values()).isoformat(),
        last_timestamp_utc=max(frame.index.max() for frame in payload.values()).isoformat(),
        alignment={
            "common_rows": int(len(common_index)),
            "dropped_bid_rows_to_align": dropped_bid,
            "dropped_ask_rows_to_align": dropped_ask,
            "fully_aligned": dropped_bid == 0 and dropped_ask == 0,
        },
    )
    _write_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Descarga automaticamente Dukascopy M1 BID+ASK y deja los CSV crudos listos para high_precision_import."
    )
    parser.add_argument("--pair", default="EURUSD")
    parser.add_argument("--start", required=True, help="Fecha inicial ISO, por ejemplo 2024-10-01")
    parser.add_argument("--end", required=True, help="Fecha final ISO inclusive, por ejemplo 2025-03-31")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HIGH_PRECISION_RAW_DIR)
    parser.add_argument("--chunk-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.5)
    args = parser.parse_args()

    manifest = download_m1_bid_ask(
        pair=args.pair.upper().strip(),
        start_date=date.fromisoformat(args.start),
        end_date=date.fromisoformat(args.end),
        output_dir=args.output_dir,
        chunk_retries=args.chunk_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
