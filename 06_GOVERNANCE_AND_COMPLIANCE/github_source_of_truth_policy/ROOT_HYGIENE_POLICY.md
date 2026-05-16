# ROOT HYGIENE POLICY

## 1. Objetivo
Mantener la raíz del proyecto libre de archivos temporales, logs sueltos, scripts de prueba o cualquier artefacto que no pertenezca a la estructura institucional de 8 carpetas.

## 2. Estructura Permitida
La raíz solo debe contener:
- `01_CORE_PRODUCTION`
- `02_INCUBATION_STAGING`
- `03_RESEARCH_LAB`
- `04_INFRASTRUCTURE_ENGINEERING`
- `05_MARKET_DATA_VAULT`
- `06_GOVERNANCE_AND_COMPLIANCE`
- `07_BACKUPS`
- `08_CLOUD_FREE_RUN_LAB`
- `.gitignore`
- `.git` (Carpeta oculta de sistema)

Excepciones técnicas, permitidas SOLO si están documentadas en `06_GOVERNANCE_AND_COMPLIANCE/root_hygiene/` (estado canónico actual): `.github`, `README.md`, `requirements.txt`, `requirements-vps-optional.txt`. `01_CORE_PRODUCTION` puede existir en disco con contenido gitignored-by-design (no trackeado).

## 3. Protocolo de Limpieza
- Prohibido dejar archivos `.zip`, `.csv`, `.py` o `.txt` sueltos en la raíz.
- Todo nuevo artefacto debe ser creado dentro de la subcarpeta correspondiente según su naturaleza (Research, Governance, Infra, etc.).
- Las carpetas temporales como `temp_zip_extract` deben ser eliminadas inmediatamente después de cumplir su función.
