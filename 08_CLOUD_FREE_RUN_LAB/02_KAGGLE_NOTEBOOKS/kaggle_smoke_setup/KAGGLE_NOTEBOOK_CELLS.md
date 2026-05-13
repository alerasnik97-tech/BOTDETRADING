# KAGGLE_NOTEBOOK_CELLS

Copia y pega las siguientes celdas en tu Kaggle Notebook.

### IMPORTANTE
- **No correr backtests**
- **No correr sweeps**
- **No conectar broker**
- **No subir datos privados sin dataset privado**
- **No imprimir secretos**

---

#### CELDA 1 — Environment check:

```python
import os, sys, platform, subprocess, pathlib, json, hashlib, time
print("Python:", sys.version)
print("Platform:", platform.platform())
print("CWD:", os.getcwd())
print("Kaggle working exists:", os.path.exists("/kaggle/working"))
print("Kaggle input exists:", os.path.exists("/kaggle/input"))
```

#### CELDA 2 — Safe repo clone using Kaggle Secret:
*Nota: Requiere un Secret llamado `GH_TOKEN` configurado en el Notebook.*

```python
import os, pathlib, subprocess, stat
from kaggle_secrets import UserSecretsClient

OWNER = "alerasnik97-tech"
REPO = "bottrading"

try:
    user_secrets = UserSecretsClient()
    token = user_secrets.get_secret("GH_TOKEN")

    netrc = pathlib.Path.home() / ".netrc"
    netrc.write_text(f"machine github.com\nlogin x-access-token\npassword {token}\n")
    netrc.chmod(0o600)

    target = "/kaggle/working/bottrading"
    if not os.path.exists(target):
        subprocess.run(["git", "clone", f"https://github.com/{OWNER}/{REPO}.git", target], check=True)
    else:
        print("Repo already exists:", target)

    os.chdir(target)
    print("Repo cloned safely. Current dir:", os.getcwd())
except Exception as e:
    print("Error during safe clone. Check GH_TOKEN secret:", e)
```

#### CELDA 3 — If repo clone is not used, inspect attached Kaggle input:

```python
import os
print("Inspeccionando /kaggle/input...")
for root, dirs, files in os.walk("/kaggle/input"):
    level = root.replace("/kaggle/input", "").count(os.sep)
    if level <= 2:
        print(root, "files:", len(files))
```

#### CELDA 4 — Project smoke inspection:

```python
import os, pathlib
base = pathlib.Path("/kaggle/working/bottrading")
print("Base path exists:", base.exists())
if base.exists():
    for p in [
        "08_CLOUD_FREE_RUN_LAB",
        "03_RESEARCH_LAB",
        "06_GOVERNANCE_AND_COMPLIANCE",
    ]:
        print(f"Folder {p}:", (base / p).exists())
else:
    print("WARNING: bottrading folder not found in /kaggle/working/")
```

#### CELDA 5 — Create Kaggle output folder:

```python
import os, json, pathlib, time
out = pathlib.Path("/kaggle/working/kaggle_smoke_outputs")
out.mkdir(exist_ok=True)
manifest = {
    "status": "KAGGLE_SMOKE_READY",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "purpose": "environment smoke test only; no strategy run",
}
(out / "KAGGLE_SMOKE_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print("Output written to:", out)
```

#### CELDA 6 — No secret cleanup:

```python
import pathlib
netrc = pathlib.Path.home() / ".netrc"
if netrc.exists():
    netrc.unlink()
    print("Removed .netrc (Security Cleanup OK)")
else:
    print("No .netrc found (Already clean)")
```

#### CELDA 7 — Final zip outputs for download:

```python
import shutil, pathlib, os
src = pathlib.Path("/kaggle/working/kaggle_smoke_outputs")
dst = pathlib.Path("/kaggle/working/KAGGLE_SMOKE_OUTPUTS")
if src.exists():
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print("Ready output folder for download:", dst)
else:
    print("Error: Source output folder not found.")
```
