# Telegram Access Revocation Cleanup — Governance Docs

**Provenance:** curated docs-only port from `clean-sync-branch` commit
`66768383` (`[security] revoke and mask exposed telegram token`).
clean-sync-branch is a historical donor / unrelated orphan branch — these
files were ported file-level (no merge, no cherry-pick).

**Path/name sanitization:** the canonical `.gitignore` intentionally ignores
any path containing `token`/`secret`/`credential`/`password` (secret-hygiene,
defense-in-depth). The original donor path
`security/token_revocation_cleanup_telegram/` and the original filenames were
therefore re-homed to this **neutral, policy-descriptive** path/names so the
governance knowledge can be tracked **without weakening, negating, or
force-adding past the `.gitignore`** (the ignore rules remain fully intact and
unchanged).

**Content:** policy / decision / confirmation / lockdown narratives only.
Secret-scanned — no real tokens, API keys, passwords, chat IDs, private keys,
or secret values (the only token-shaped string is an explicit illustrative
masked example inside the scan policy).

**Sensitive audit artifacts intentionally excluded** (NOT ported — owner
review only): `*.csv` (`SECRET_SCAN_AFTER_REVOCATION_MASKED.csv`,
`TOKEN_EXPOSURE_FILES_AUDIT.csv`, `TOKEN_MASKING_ACTIONS.csv`), `*.txt`
(`TOKEN_CLEANUP_GIT_STATUS.txt`), and `PRE_COMMIT_SECRET_SCAN_PLAN.md`
(regex false-positive — conservatively excluded).

**Key facts preserved:** the exposed Telegram bot token was revoked by the
owner via @BotFather (2026-05-14); the new token is NOT in the repository and
must live only in a local env var / secret manager; git *history* still
contains the old (revoked, inactive) token — a separate, owner-authorized
history-purge phase is required and was NOT performed here.
