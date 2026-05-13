# OWNERSHIP RULES — 05_MARKET_DATA_VAULT

**Owner:** Data Quality Agent
**Write:** Data Quality Agent
**Read:** Research, Governance, ChatGPT Audit
**Purpose:** Datos READ-ONLY. Ticks, parquet, historicos.
**Prohibitions:** No sobrescribir, no Git para datos crudos, no empaquetar crudos.
**Git:** Commit local OK, push requires user approval.
**ZIP:** Per-folder export in _CHATGPT_EXPORT/. Master ZIP is 000_PARA_CHATGPT.zip.
**Escalation:** Report anomalies to 06_GOVERNANCE. Block if uncertain.
