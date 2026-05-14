# KAGGLE_V49_7C_NOTEBOOK_CELLS

Copia estas celdas en tu Notebook de Kaggle para la corrida V49.7C.

---

### CELDA 1 — Verificar entorno Python

```python
import os, sys, platform, subprocess, pathlib, json, hashlib, time
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"CWD: {os.getcwd()}")
```

---

### CELDA 2 — Clonar repo branch clean-sync-branch

```python
import os, pathlib, subprocess
from kaggle_secrets import UserSecretsClient

OWNER = "alerasnik97-tech"
REPO = "bottrading"
BRANCH = "clean-sync-branch"

user_secrets = UserSecretsClient()
token = user_secrets.get_secret("GH_TOKEN")

netrc = pathlib.Path.home() / ".netrc"
netrc.write_text(f"machine github.com\nlogin x-access-token\npassword {token}\n")
netrc.chmod(0o600)

target = "/kaggle/working/bottrading"
if not os.path.exists(target):
    subprocess.run(["git", "clone", "-b", BRANCH, f"https://github.com/{OWNER}/{REPO}.git", target], check=True)
else:
    print("Repo ya existe.")

os.chdir(target)
print(f"Clonado exitoso en: {os.getcwd()}")
```

---

### CELDA 3 — Instalar dependencias mínimas

```python
!pip install -r requirements.txt
# !pip install pandas numpy matplotlib seaborn scipy  # Si no están en requirements
```

---

### CELDA 4 — Verificar branch y limpieza de secretos

```python
!git branch --show-current
netrc = pathlib.Path.home() / ".netrc"
if netrc.exists():
    netrc.unlink()
    print("Credenciales de GitHub eliminadas del disco.")
```

---

### CELDA 5 — Engine Verify (Opcional)

```python
# !pytest src/v7_engine/tests/ --maxfail=5
```

---

### CELDA 6 — Ejecutar Preflight Cloud-Safe

```python
# Carga de la configuración desde el archivo creado por Antigravity
import json
config_path = "08_CLOUD_FREE_RUN_LAB/02_KAGGLE_NOTEBOOKS/v49_7c_full_scope_runner/KAGGLE_V49_7C_RUN_CONFIG.json"
with open(config_path, "r") as f:
    run_config = json.load(f)

print("Configuración cargada para V49.7C:")
print(json.dumps(run_config, indent=2))

# AQUÍ IRÍA LA LLAMADA AL PREFLIGHT DEL MOTOR
# run_preflight(run_config)
```

---

### CELDA 7 — Ejecutar V49.7C (Mañana tras autorización)

```python
# NO EJECUTAR HASTA QUE SE TERMINE LA CONFIGURACIÓN REAL
# run_sweep(run_config)
```

---

### CELDA 8 — Guardar outputs y generar manifest

```python
import shutil
output_dir = "/kaggle/working/v49_7c_outputs"
os.makedirs(output_dir, exist_ok=True)

# Simulación de generación de manifest
manifest = {
    "run_id": "V49_7C_KAGGLE_001",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "files": os.listdir(output_dir)
}
with open(f"{output_dir}/manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Outputs preparados en: {output_dir}")
```
