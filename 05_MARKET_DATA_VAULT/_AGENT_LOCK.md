# _AGENT_LOCK — 05_MARKET_DATA_VAULT

WRITE_ALLOWED_FOR: Data Quality Agent
READ_ALLOWED_FOR: Research, Governance, ChatGPT Audit
NEVER_TOUCH: .git, raw market data, backups, production releases (unless authorized)
MUST_REPORT: FILES_CHANGED.csv, COMMANDS_RUN.txt, TEST_OUTPUT.txt, GIT_STATUS.txt, FINAL_REPORT.md
STOP_CONDITIONS: test failure, unexpected data mutation, unknown file movement, Git dirty state
