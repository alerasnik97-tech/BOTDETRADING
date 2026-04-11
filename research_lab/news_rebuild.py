from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_lab.config import DEFAULT_NEWS_FILE, DEFAULT_PAIR, DEFAULT_RAW_NEWS_FILE, NewsConfig
from research_lab.news_filter import build_news_datasets, load_news_events, news_result_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruye el dataset limpio de noticias para el proyecto.")
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--raw-file", default=str(DEFAULT_RAW_NEWS_FILE))
    parser.add_argument("--clean-file", default=str(DEFAULT_NEWS_FILE))
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--currencies", nargs="*", default=["USD", "EUR"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = NewsConfig(
        enabled=True,
        file_path=Path(args.clean_file),
        raw_file_path=Path(args.raw_file),
        pre_minutes=15,
        post_minutes=15,
        currencies=tuple(args.currencies),
    )
    clean_frame, audit_frame, diagnostics = build_news_datasets(args.pair.upper().strip(), settings, start=args.start, end=args.end)
    result = load_news_events(args.pair.upper().strip(), settings)
    print(
        json.dumps(
            {
                "clean_rows": len(clean_frame),
                "audit_rows": len(audit_frame),
                "result": news_result_payload(result),
                "diagnostics": diagnostics,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
