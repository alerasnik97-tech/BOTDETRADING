from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from research_lab.config import DEFAULT_HIGH_PRECISION_PREPARED_DIR, DEFAULT_HIGH_PRECISION_RAW_DIR, NY_TZ
from research_lab.data_loader import OHLCV_COLUMNS, parse_prepared_index, validate_price_frame


SUPPORTED_HIGH_PRECISION_SOURCE_TYPES = ("dukascopy_m1_bid_ask", "dukascopy_tick_bid_ask")
TICK_COLUMNS = ("price", "volume")


def _read_ohlcv_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = parse_prepared_index(frame.index)
    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} no contiene columnas OHLCV requeridas: {missing}")
    frame = frame[list(OHLCV_COLUMNS)].astype(float)
    validate_price_frame(frame)
    return frame


def _read_tick_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = parse_prepared_index(frame.index)
    if "price" not in frame.columns:
        raise ValueError(f"{path} no contiene la columna 'price'.")
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame = frame[["price", "volume"]].copy()
    frame["price"] = frame["price"].astype(float)
    frame["volume"] = frame["volume"].astype(float)
    if frame.index.duplicated().any():
        frame = frame.groupby(level=0).agg({"price": "last", "volume": "sum"}).sort_index()
    if not frame.index.is_monotonic_increasing:
        frame = frame.sort_index()
    return frame


def _resample_ticks_to_m1(frame: pd.DataFrame) -> pd.DataFrame:
    resampled = (
        frame.resample("1min", label="right", closed="right")
        .agg({"price": ["first", "max", "min", "last"], "volume": "sum"})
        .dropna()
    )
    resampled.columns = ["open", "high", "low", "close", "volume"]
    validate_price_frame(resampled)
    return resampled


def _combine_mid_from_bid_ask(bid: pd.DataFrame, ask: pd.DataFrame) -> pd.DataFrame:
    common_index = bid.index.intersection(ask.index)
    if common_index.empty:
        raise ValueError("BID y ASK no comparten timestamps; no se puede derivar MID.")
    bid = bid.loc[common_index]
    ask = ask.loc[common_index]
    mid = pd.DataFrame(index=common_index)
    for column in ("open", "high", "low", "close"):
        mid[column] = (bid[column].astype(float) + ask[column].astype(float)) / 2.0
    mid["volume"] = (bid["volume"].astype(float) + ask["volume"].astype(float)) / 2.0
    validate_price_frame(mid)
    return mid


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)


def _source_files_for(pair: str, source_type: str, input_dir: Path) -> dict[str, Path]:
    pair = pair.upper().strip()
    if source_type == "dukascopy_m1_bid_ask":
        return {
            "bid": input_dir / f"{pair}_M1_BID.csv",
            "ask": input_dir / f"{pair}_M1_ASK.csv",
        }
    if source_type == "dukascopy_tick_bid_ask":
        return {
            "bid": input_dir / f"{pair}_TICK_BID.csv",
            "ask": input_dir / f"{pair}_TICK_ASK.csv",
        }
    raise ValueError(f"source_type no soportado: {source_type}")


def expected_input_schema(source_type: str) -> dict[str, Any]:
    if source_type == "dukascopy_m1_bid_ask":
        return {
            "source_type": source_type,
            "expected_files": ["EURUSD_M1_BID.csv", "EURUSD_M1_ASK.csv"],
            "index_requirement": "primer campo = timestamp con timezone explicito, por ejemplo 2025-01-02 11:00:00-05:00",
            "required_columns": list(OHLCV_COLUMNS),
            "recommended_target_dir": str(DEFAULT_HIGH_PRECISION_RAW_DIR),
        }
    if source_type == "dukascopy_tick_bid_ask":
        return {
            "source_type": source_type,
            "expected_files": ["EURUSD_TICK_BID.csv", "EURUSD_TICK_ASK.csv"],
            "index_requirement": "primer campo = timestamp con timezone explicito",
            "required_columns": list(TICK_COLUMNS),
            "recommended_target_dir": str(DEFAULT_HIGH_PRECISION_RAW_DIR),
        }
    raise ValueError(f"source_type no soportado: {source_type}")


def integrate_high_precision_source(
    *,
    pair: str,
    source_type: str,
    input_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_type = source_type.strip().lower()
    if source_type not in SUPPORTED_HIGH_PRECISION_SOURCE_TYPES:
        raise ValueError(f"source_type no soportado: {source_type}")

    files = _source_files_for(pair, source_type, input_dir)
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos reales para integrar la fuente de alta precision. "
            f"Esperados: {missing}"
        )

    if source_type == "dukascopy_m1_bid_ask":
        bid = _read_ohlcv_csv(files["bid"])
        ask = _read_ohlcv_csv(files["ask"])
    else:
        bid = _resample_ticks_to_m1(_read_tick_csv(files["bid"]))
        ask = _resample_ticks_to_m1(_read_tick_csv(files["ask"]))

    common_index = bid.index.intersection(ask.index)
    if common_index.empty:
        raise ValueError("Los archivos BID y ASK no comparten timestamps.")
    bid = bid.loc[common_index].copy()
    ask = ask.loc[common_index].copy()
    mid = _combine_mid_from_bid_ask(bid, ask)

    output_dir.mkdir(parents=True, exist_ok=True)
    bid_out = output_dir / f"{pair}_M1_BID.csv"
    ask_out = output_dir / f"{pair}_M1_ASK.csv"
    mid_out = output_dir / f"{pair}_M1_MID.csv"
    _write_frame(bid_out, bid)
    _write_frame(ask_out, ask)
    _write_frame(mid_out, mid)

    manifest = {
        "provider": "dukascopy",
        "source_type": source_type,
        "pair": pair,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_files": {key: str(value) for key, value in files.items()},
        "generated_files": {
            "bid_m1": str(bid_out),
            "ask_m1": str(ask_out),
            "mid_m1": str(mid_out),
        },
        "timezone": NY_TZ,
        "rows_bid": int(len(bid)),
        "rows_ask": int(len(ask)),
        "rows_mid": int(len(mid)),
        "first_timestamp_ny": str(common_index.min()),
        "last_timestamp_ny": str(common_index.max()),
        "notes": [
            "mid_m1 es derivado como promedio de bid y ask reales",
            "la fuente puede activarse desde high_precision_mode mediante el loader del laboratorio",
            "la precision intrabar mejora a resolucion M1, pero no reemplaza tick data real",
        ],
    }
    manifest_path = output_dir / f"{pair}_high_precision_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Integra una fuente Dukascopy bid+ask real al formato canonico de alta precision del proyecto.")
    parser.add_argument("--pair", required=False, default="EURUSD")
    parser.add_argument("--source-type", required=False, choices=SUPPORTED_HIGH_PRECISION_SOURCE_TYPES)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_HIGH_PRECISION_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HIGH_PRECISION_PREPARED_DIR)
    parser.add_argument("--print-schema", action="store_true")
    args = parser.parse_args()

    if args.print_schema:
        if not args.source_type:
            raise ValueError("--print-schema requiere --source-type")
        print(json.dumps(expected_input_schema(args.source_type), indent=2, ensure_ascii=False))
        return

    if not args.source_type:
        raise ValueError("--source-type es obligatorio salvo que uses --print-schema")

    manifest = integrate_high_precision_source(
        pair=args.pair.upper().strip(),
        source_type=args.source_type,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
