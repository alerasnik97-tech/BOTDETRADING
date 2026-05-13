# _AGENT_LOCK — 06_GOVERNANCE_AND_COMPLIANCE

WRITE_ALLOWED_FOR: Governance Agent
READ_ALLOWED_FOR: ALL
NEVER_TOUCH: .git, raw market data, backups, production releases (unless authorized)
MUST_REPORT: FILES_CHANGED.csv, COMMANDS_RUN.txt, TEST_OUTPUT.txt, GIT_STATUS.txt, FINAL_REPORT.md
STOP_CONDITIONS: test failure, unexpected data mutation, unknown file movement, Git dirty state
