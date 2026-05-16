# NEXT PROMPT — EXPLICIT DATA REGEN AUTHORIZATION (owner sign-off)

**Use ONLY if the owner cannot supply a file from backup and chooses to
authorize regeneration. Default remains: audit only / owner-supplied.**

Regeneration is **forbidden without explicit, per-asset, written owner
authorization** because the relevant pipelines are network-capable and span
the prohibited 2025/2026 period.

## Per-asset authorization (owner fills EXPLICITLY — one decision each)

| asset | regen pipeline (if any) | network? | spans 2025/26? | owner decision |
|---|---|---|---|---|
| `forex_factory_cache.csv` | (Forex Factory cache builder — confirm exact entrypoint) | likely YES | likely YES | ☐ SUPPLY FROM BACKUP ☐ AUTHORIZE REGEN ☐ KEEP BLOCKED |
| `news_eurusd_v2_utc.csv` | `build_am_grade_news_dataset` / news rebuild | depends | depends | ☐ SUPPLY ☐ AUTHORIZE REGEN ☐ KEEP BLOCKED |
| hi-precision M1 dukascopy | dukascopy M1 acquisition | YES (dukascopy) | YES | ☐ SUPPLY ☐ AUTHORIZE REGEN ☐ KEEP BLOCKED |
| `canonical_anchor_events.csv` | `research_lab/official_anchors` (BLS `urllib` / stubs) | YES if live connectors | YES (`years=[2024,2025,2026]`) | ☐ ACCEPT AS-IS (provenance waiver) ☐ REGEN UNDER POLICY ☐ QUARANTINE |

## Conditions if regen is authorized (per asset)
1. Owner states explicitly, in writing, which asset and which mode
   (offline `stubs` vs live network). Live network requires a separate,
   explicit "internet authorized for asset X" statement.
2. Run logged: connector used, network on/off, date range, sha256, row count
   → recorded in `06_GOVERNANCE_AND_COMPLIANCE/data_governance/`.
3. Existing files are never overwritten without hash comparison + backup.
4. Regeneration is a **separate authorized phase**, NOT part of any audit.

## Hard rules (unchanged)
- No regen/scrape/download/synthesis without the explicit per-asset sign-off
  above. No 2025/2026 *analysis*. No backtest/strategy/F06-real/optimization/
  sweep/validation/holdout. No engine edits. No `main`, no force push, no ZIP.
  Never fabricate or stub data to pass tests.
