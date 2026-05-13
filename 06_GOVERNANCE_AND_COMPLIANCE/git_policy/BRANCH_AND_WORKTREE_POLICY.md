# BRANCH AND WORKTREE POLICY

## Branches
- `main` — clean baseline, no direct work except minor authorized tasks
- `agent/research-*` — research and gate work
- `agent/infra-*` — packaging, maintenance
- `agent/governance-*` — architecture, rules
- `agent/data-*` — data quality
- `agent/production-*` — releases
- `agent/incubation-*` — forward testing

## Worktree Locations (proposed, not yet created)
- `C:\Users\alera\Desktop\Bot\WORKTREES\<branch_name>`

## Rules
- No two agents write to same directory simultaneously
- Before merge: tests + manifest + git status + audit
- No push without user authorization
- No rebase on main
