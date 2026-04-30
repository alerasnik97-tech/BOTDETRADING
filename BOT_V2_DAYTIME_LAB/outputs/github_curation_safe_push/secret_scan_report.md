# Secret Scan Report

- timestamp_utc: 2026-04-30T11:18:50.374823+00:00
- scan_scope: candidate versionable files <=2MB plus sensitive path names; dependency/cache trees excluded by policy
- candidate_files_scanned: 1852
- pass: True
- path keyword findings: 16
- reviewed content hits: 468
- real findings: 0
- action: commit_allowed

## Result
No high-confidence real secret value was detected in the versionable candidate set. Sensitive/local-only paths remain excluded from Git.

## False Positive / Excluded Notes
- `.pkg/packaging/_tokenizer.py` - dependency/cache naming hit; excluded from Git
- `.vendor_duka2/markdown_it/token.py` - dependency/cache naming hit; excluded from Git
- `.vendor_duka2/polars/io/cloud/credential_provider/__init__.py` - dependency/cache naming hit; excluded from Git
- `.vendor_duka2/polars/io/cloud/credential_provider/_builder.py` - dependency/cache naming hit; excluded from Git
- `.vendor_duka2/polars/io/cloud/credential_provider/_providers.py` - dependency/cache naming hit; excluded from Git
- `.vendor_duka2/pygments/token.py` - dependency/cache naming hit; excluded from Git
- `.venv_fixed/Lib/site-packages/packaging/_tokenizer.py` - dependency/cache naming hit; excluded from Git
- `.venv_fixed/Lib/site-packages/pip/_vendor/packaging/_tokenizer.py` - dependency/cache naming hit; excluded from Git
- `.venv_fixed/Lib/site-packages/pip/_vendor/pygments/token.py` - dependency/cache naming hit; excluded from Git
- `.venv_fixed/Lib/site-packages/pygments/token.py` - dependency/cache naming hit; excluded from Git
- `BOT_V2_DAYTIME_LAB/outputs/phase36r_37a_micro_real_gate/exness_symbol_gate/phase36r_exness_symbol_gate.json` - keyword/name risk reviewed; no high-confidence secret value detected
- `BOT_V2_DAYTIME_LAB/outputs/phase36r_37a_micro_real_gate/exness_symbol_gate/phase36r_exness_symbol_gate.md` - keyword/name risk reviewed; no high-confidence secret value detected
- `BOT_V2_DAYTIME_LAB/src/__pycache__/phase36_exness_lot_validator.cpython-314.pyc` - keyword/name risk reviewed; no high-confidence secret value detected
- `BOT_V2_DAYTIME_LAB/src/__pycache__/phase36_exness_symbol_gate.cpython-314.pyc` - keyword/name risk reviewed; no high-confidence secret value detected
- `BOT_V2_DAYTIME_LAB/src/phase36_exness_lot_validator.py` - keyword/name risk reviewed; no high-confidence secret value detected
- `BOT_V2_DAYTIME_LAB/src/phase36_exness_symbol_gate.py` - keyword/name risk reviewed; no high-confidence secret value detected
- `BOT_V2_DAYTIME_LAB/src/phase37d_live_news_api_adapter.py:150` - code/template false positive for assignment_api_token: return f"https://financialmodelingprep.com/api/v3/economic_calendar?from={s}&to={e}&apikey=***REDACTED***"
- `legacy/root_scripts/fx_multi_timeframe_backtester.py:4047` - code/template false positive for assignment_api_token: news_api_key=***REDACTED***,
- `VPS_READINESS/scripts/vps_mt5_connection_check.py:28` - code/template false positive for assignment_password: password=***REDACTED***"password"),
