# _AGENT_LOCK — 02_INCUBATION_STAGING

WRITE_ALLOWED_FOR: Incubation Agent
READ_ALLOWED_FOR: Governance, ChatGPT Audit
NEVER_TOUCH: .git, raw market data, backups, production releases (unless authorized)
MUST_REPORT: FILES_CHANGED.csv, COMMANDS_RUN.txt, TEST_OUTPUT.txt, GIT_STATUS.txt, FINAL_REPORT.md
STOP_CONDITIONS: test failure, unexpected data mutation, unknown file movement, Git dirty state
