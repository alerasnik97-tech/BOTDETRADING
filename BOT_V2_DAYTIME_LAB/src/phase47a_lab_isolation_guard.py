import subprocess
import sys
from pathlib import Path


PASS_VERDICT = "LAB_ISOLATION_GUARD_PASS"
FAIL_VERDICT = "LAB_ISOLATION_GUARD_FAIL_PROTECTED_CHANGE"


PROTECTED_PREFIXES = (
    "MANIPULANTE/",
    "mt5_demo_executor_lab/",
    "ROCKI_AM/",
)

PROTECTED_ROOT_FILES = {
    "START_MANIPULANTE.bat",
    "STOP_MANIPULANTE.bat",
    "STATUS_MANIPULANTE.bat",
}

ALLOWED_PREFIXES = (
    "LAB_STRATEGIES/",
)

ALLOWED_PHASE47A_FILES = {
    "PROJECT_ZONES_AND_BRANCHING_RULES.md",
    "BOT_V2_DAYTIME_LAB/src/phase47a_lab_isolation_guard.py",
    "BOT_V2_DAYTIME_LAB/reports/PHASE47A_LAB_ISOLATION_AND_MANIPULANTE_PROTECTION_REPORT.md",
    "BOT_V2_DAYTIME_LAB/reports/PHASE47A_LAB_ISOLATION_AND_MANIPULANTE_PROTECTION_REPORT.json",
    "BOT_V2_DAYTIME_LAB/reports/PHASE47H_PHASE47A_COMMIT_READINESS_REPORT.md",
    "BOT_V2_DAYTIME_LAB/reports/PHASE47H_PHASE47A_COMMIT_READINESS_REPORT.json",
}


def run_git(args, root=None, check=True):
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(stderr or f"git {' '.join(args)} failed")
    return completed.stdout


def find_repo_root():
    output = run_git(["rev-parse", "--show-toplevel"])
    return Path(output.strip()).resolve()


def current_branch(root):
    branch = run_git(["branch", "--show-current"], root=root, check=False).strip()
    if branch:
        return branch
    commit = run_git(["rev-parse", "--short", "HEAD"], root=root, check=False).strip()
    return f"DETACHED@{commit}" if commit else "UNKNOWN"


def parse_porcelain_line(line):
    status = line[:2]
    raw_path = line[3:].strip()
    if " -> " in raw_path:
        raw_path = raw_path.split(" -> ", 1)[1]
    return status, raw_path.strip('"')


def normalize_path(path_text):
    return path_text.replace("\\", "/").lstrip("./")


def is_phase_file(path_lower):
    return (
        path_lower.startswith("bot_v2_daytime_lab/src/phase37")
        or path_lower.startswith("bot_v2_daytime_lab/src/phase44")
        or path_lower.startswith("bot_v2_daytime_lab/src/phase45")
        or path_lower.startswith("bot_v2_daytime_lab/src/phase46")
    )


def is_secret_or_live_config(path_lower):
    sensitive_tokens = (
        ".env",
        "secret",
        "token",
        "credential",
        "api_key",
        "private_key",
        "config.local",
        "alerts_config.local",
        "live_config",
        "/live/",
        "live_news",
        "news_fortress",
        "data_quality_mask",
        "certified_m3",
        "stop_bot",
        "orders",
        "order_router",
    )
    return any(token in path_lower for token in sensitive_tokens)


def classify_path(path_text):
    path = normalize_path(path_text)
    lower = path.lower()
    basename = Path(path).name

    if path in PROTECTED_ROOT_FILES or basename in PROTECTED_ROOT_FILES:
        return "PROTECTED_CHANGE"
    if any(path.startswith(prefix) for prefix in PROTECTED_PREFIXES):
        return "PROTECTED_CHANGE"
    if is_phase_file(lower):
        return "PROTECTED_CHANGE"
    if is_secret_or_live_config(lower):
        return "PROTECTED_CHANGE"
    if "scbi_m5_global" in lower:
        return "PROTECTED_CHANGE"

    if any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        return "ALLOWED_LAB_CHANGE"
    if path in ALLOWED_PHASE47A_FILES:
        return "ALLOWED_DOC_CHANGE"
    if path.startswith("BOT_V2_DAYTIME_LAB/reports/PHASE47A_"):
        return "ALLOWED_DOC_CHANGE"
    if path.startswith("BOT_V2_DAYTIME_LAB/reports/PHASE47H_"):
        return "ALLOWED_DOC_CHANGE"

    return "UNKNOWN_CHANGE"


def get_status_entries(root, staged_only=False):
    args = ["status", "--porcelain"]
    output = run_git(args, root=root)
    entries = []
    for line in output.splitlines():
        if not line:
            continue
        status, path = parse_porcelain_line(line)
        
        # In staged_only mode, we only care about the first column (index)
        # X is status of the index, Y is status of the work tree
        # If X is ' ' or '?', it's not staged.
        if staged_only and status[0] in (" ", "?"):
            continue
            
        entries.append(
            {
                "status": status,
                "path": normalize_path(path),
                "classification": classify_path(path),
            }
        )
    return entries


def print_report(root, branch, entries, staged_only):
    mode = "STAGED-ONLY" if staged_only else "FULL WORKTREE"
    print(f"PHASE47A LAB ISOLATION GUARD ({mode})")
    print(f"repo_root: {root}")
    print(f"branch: {branch}")
    print(f"changes_detected: {len(entries)}")
    print("")

    if not entries:
        print("No relevant git changes detected.")
        print(f"verdict: {PASS_VERDICT}")
        return

    protected = 0
    for entry in entries:
        print(f"{entry['classification']} | {entry['status']} | {entry['path']}")
        if entry["classification"] == "PROTECTED_CHANGE":
            protected += 1

    print("")
    if protected:
        print(f"protected_changes: {protected}")
        print(f"verdict: {FAIL_VERDICT}")
    else:
        print("protected_changes: 0")
        print(f"verdict: {PASS_VERDICT}")


def main():
    try:
        staged_only = "--staged-only" in sys.argv
        root = find_repo_root()
        branch = current_branch(root)
        entries = get_status_entries(root, staged_only=staged_only)
        print_report(root, branch, entries, staged_only)
        has_protected = any(e["classification"] == "PROTECTED_CHANGE" for e in entries)
        return 1 if has_protected else 0
    except Exception as exc:
        print("PHASE47A LAB ISOLATION GUARD")
        print(f"error: {exc}")
        print(f"verdict: {FAIL_VERDICT}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
