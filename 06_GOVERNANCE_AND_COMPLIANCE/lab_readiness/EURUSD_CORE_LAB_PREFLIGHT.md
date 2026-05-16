# EURUSD CORE LAB PREFLIGHT

Date: 2026-05-16
Branch: governance/phase-d-reconciliation-20260516
Base commit audited: 62901434c4e439e87f1892fb6c9bdd82111c2948
Mode: report-only. No backtest, no strategy run, no F06 real run, no validation, no holdout.

Intended next lab scope: EURUSD intraday research, 07:00-19:00 New York, max 3 trades/day, train-only, no 2025/2026 holdout analysis.

## Preflight Checks

| Check | Evidence | Verdict |
|---|---|---|
| Branch | `git branch --show-current` = `governance/phase-d-reconciliation-20260516` | PASS |
| Local HEAD | `62901434c4e439e87f1892fb6c9bdd82111c2948`; origin same after fetch | PASS |
| Main touched | branch is not `main` | PASS |
| Active Python process | `Get-Process python` and WMI command returned no active python.exe process | PASS |
| Root structure | 8 canonical folders plus documented technical exceptions `.github`, `README.md`, `requirements*.txt`; no loose root scripts/reports/data | PASS_WITH_DOCUMENTED_EXCEPTIONS |
| No ZIP inside repo | one ignored legacy ZIP was found under `07_BACKUPS/old_deliveries/` and moved to `C:\Users\alera\Desktop\Bot\BOT_ZIP_LEGACY_ARCHIVE\`; `rg --files -g "*.zip"` now returns none | PASS_AFTER_MOVE |
| DATA_ROOT policy | configured data paths point under `05_MARKET_DATA_VAULT` | PASS |
| EURUSD prepared train data | `DEFAULT_DATA_DIRS` = `05_MARKET_DATA_VAULT/eurusd_data/*/prepared`; both prepared dirs are empty | FAIL_HARD_BLOCKER |
| Raw EURUSD material | `05_MARKET_DATA_VAULT/BOT_MARKET_DATA/tick/EURUSD` exists: 3548 files, 9,210,754,686 bytes; monthly parquet count 137 | PRESENT_BUT_NOT_WIRED |
| Loader contract | `research_lab.data_loader.load_prepared_ohlcv` reads prepared CSV files named `EURUSD_<TF>.csv`; it does not read the raw tick tree | FAIL_CANNOT_LAB_WITHOUT_OWNER_DATA_DECISION |
| Holdout untouched | no backtest/validation/holdout process run; only file inventory by name/size/hash | PASS |
| Strategy registry import | `from research_lab.strategies import STRATEGY_REGISTRY` returns 63 entries | PASS |
| Engine/cost import | `from research_lab.config import EngineConfig, NewsConfig; import research_lab.engine` | PASS |
| `research_lab` import | `research_lab OK` | PASS |
| `light_runner` import | `light_runner OK` after compatibility aliases | PASS |
| `audit_level3` import | `audit_level3 OK` after `DEFAULT_NEWS_SUMMARY_FILE` compatibility alias | PASS |
| Cost model config | `EngineConfig` imports and default config is available | PASS |
| Output policy | root ZIP workflow forbidden; reports belong under governance/research folders | PASS |
| Optional modules isolated | `forex_factory_cache.csv`, `news_eurusd_v2_utc.csv`, hi-precision dukascopy, and anchor provenance are not required for the EURUSD prepared-OHLCV core path | PASS_WITH_SCOPE_LIMIT |
| F06 pipeline tests | 119/119 pass in final safe run | PASS |
| Broader research_lab tests | 164 run, 16 failures, 9 errors, 13 skipped after cleanup | FAIL_REMAINING_NON_GREEN_SUITE |

## Verdict

EURUSD core lab is not authorized.

The listed missing assets (`forex_factory_cache.csv`, `news_eurusd_v2_utc.csv`, hi-precision dukascopy, and unverified anchor provenance) do not directly block a train-only EURUSD prepared-OHLCV lab if news/high-precision/Jpy modules stay disabled.

The discovered hard core blocker is different and more direct: EURUSD prepared OHLCV is absent at the configured loader path. Raw tick material exists in the vault, but it is not wired into the research loader. Bridging raw ticks to prepared CSVs or repointing the loader is a research-data authority decision and was not done here.

There is also a remaining pre-lab engineering blocker: after import/path cleanup, the broader research_lab suite still has engine/level2/high-precision contract failures. Those cannot be hidden or mass-skipped. They require a separate owner/Claude triage before the lab can be treated as clean.
