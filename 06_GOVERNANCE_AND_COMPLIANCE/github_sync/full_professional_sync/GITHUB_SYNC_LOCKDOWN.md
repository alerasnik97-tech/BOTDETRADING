# GITHUB SYNC LOCKDOWN
**Project:** BOT DE TRADING ultimo  
**Phase:** Institutional GitHub Synchronization  
**Status:** ACTIVE LOCKDOWN  

## Operational Prohibitions
- **NO** mutation of raw data, parquet, or ticks in `05_MARKET_DATA_VAULT`.
- **NO** modification of `01_CORE_PRODUCTION` or `02_INCUBATION_STAGING` binaries.
- **NO** interruption of active Manipulante 4 (M4) research.
- **NO** regeneration of `000_PARA_CHATGPT.zip`.
- **NO** inclusion of secrets, tokens, or private credentials.
- **NO** push to `main` branch.
- **NO** opening of Windows Explorer.

## Synchronization Protocol
1. **Audit:** Complete mapping of local vs. remote state.
2. **Classification:** Separation of institutional logic from heavy data.
3. **Hard Exclusion:** Reinforcement of `.gitignore` and removal of prohibited files from staging.
4. **Selective Staging:** Granular `git add` by area.
5. **Safe Push:** Upload to current working branch `agent/research-manipulante4-sweep-quality`.

---
*Certified by: Institutional Sync Agent*
