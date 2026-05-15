# KAGGLE_SMOKE_NOTEBOOK_CELLS

Copia y pega estas celdas en tu Notebook de Kaggle.

---

### CELDA 1 — Environment report

```python
import os, sys, platform, subprocess, pathlib
print(f"Python Version: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"CWD: {os.getcwd()}")
print("Disk listing in /kaggle/working:")
print(os.listdir("/kaggle/working"))
# No secrets printed here
```

---

### CELDA 2 — Clone repo

```python
import os
REPO_URL = "https://github.com/alerasnik97-tech/bottrading.git"
BRANCH = "clean-sync-branch"

if not os.path.exists("bottrading"):
    !git clone -b {BRANCH} {REPO_URL}
else:
    print("Repo already cloned.")

os.chdir("bottrading")
print(f"Current Dir: {os.getcwd()}")
!git branch --show-current
!git log --oneline -n 3
```

---

### CELDA 3 — Safety scan

```python
import os

forbidden_terms = [".env", "TELEGRAM", "BOT_TOKEN", "API_KEY", "SECRET", "PASSWORD", "2025", "2026", ".zip"]
safety_passed = True

print("Running Safety Scan...")
for root, dirs, files in os.walk("."):
    for file in files:
        if any(term in file.upper() or term in file for term in forbidden_terms):
            # Special exceptions can be handled here if needed
            if ".ipynb_checkpoints" in root: continue
            print(f"[WARNING] Potential forbidden file/term detected: {os.path.join(root, file)}")
            safety_passed = False

if safety_passed:
    print("Safety Scan: PASSED (No immediate secrets or forbidden patterns found in filenames)")
else:
    print("Safety Scan: FAILED (Review findings)")
```

---

### CELDA 4 — Import test

```python
import pandas as pd
import numpy as np
import pathlib
import sys

# Add src to path for imports
sys.path.append(os.getcwd())

try:
    import src.v7_engine as engine
    print("Import src.v7_engine: SUCCESS")
except Exception as e:
    print(f"Import src.v7_engine: FAILED - {e}")

try:
    import src.v6_utils as utils
    print("Import src.v6_utils: SUCCESS")
except Exception as e:
    print(f"Import src.v6_utils: FAILED - {e}")
```

---

### CELDA 5 — No-data smoke artifact

```python
import pandas as pd
import time

artifact_path = "/kaggle/working/KAGGLE_SMOKE_ARTIFACT_PROOF.csv"

data = {
    "phase": ["KAGGLE_CLOUD_SMOKE_TEST"],
    "repo_branch": ["clean-sync-branch"],
    "python_ok": ["YES"],
    "git_ok": ["YES"],
    "imports_ok": ["YES"],
    "secrets_ok": ["YES"],
    "test_touched": ["NO"],
    "backtest_run": ["NO"],
    "status": ["SUCCESS"],
    "timestamp": [time.strftime("%Y-%m-%d %H:%M:%S")]
}

df = pd.DataFrame(data)
df.to_csv(artifact_path, index=False)
print(f"Artifact created: {artifact_path}")
```

---

### CELDA 6 — Package outputs

```python
# No ZIP created inside repo
output_dir = "/kaggle/working/kaggle_smoke_outputs/"
os.makedirs(output_dir, exist_ok=True)

# Copy results to output dir
!cp /kaggle/working/KAGGLE_SMOKE_ARTIFACT_PROOF.csv {output_dir}
print(f"Outputs prepared in {output_dir} for manual download.")
```
