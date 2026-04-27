# GITHUB PRE-PUSH AUDIT

**Estado:** REVISIÓN REQUERIDA
**Fecha:** 2026-04-27

## 1. Lo más importante
El repositorio local está conectado a un remoto en GitHub, pero existe un desajuste masivo entre la estructura local actual (limpia y saneada) y los archivos que Git intenta rastrear. Se han detectado archivos de datos de más de 100MB y archivos de configuración con credenciales que **no deben ser subidos**.

## 2. Veredicto
**AUDIT_WARNING_STAGING_REQUIRED**
No se recomienda hacer push directo a `main` sin aplicar un `.gitignore` actualizado y una limpieza de staging.

## 3. Estado de Git
- **Rama Actual:** `main` (Adelantada por 1 commit respecto a `origin/main`).
- **Remoto:** `https://github.com/alerasnik97-tech/bottrading.git`
- **Estado de Staging:** Hay cientos de archivos "Untracked" debido a la reestructuración reciente.

## 4. Auditoría de Archivos Grandes (>10 MB)

| Archivo | Tamaño | Recomendación |
| :--- | :--- | :--- |
| `data_intake_2015_2019\...\EURUSD_M1_BID.csv` | **115.8 MB** | **IGNORAR** (Data pesada) |
| `data\forex_factory_cache.csv` | 65.1 MB | Ignorar o Git LFS |
| `ARCHIVE_SUPERSEDED\duplicated_folders\...` | >10 MB | **IGNORAR** (Basura histórica) |
| `000_PARA_CHATGPT.zip` | variable | Subir como artefacto maestro (opcional) |

## 5. Auditoría de Secretos (Riesgo Alto)
Se detectó el siguiente archivo sensible:
- `mt5_demo_executor_lab\local_launch\mt5_local_config.json`
**Acción:** Debe ser añadido a `.gitignore` inmediatamente. Contiene credenciales de acceso a MT5.

## 6. Propuesta de .gitignore Actualizado

Se recomienda añadir/modificar:
```gitignore
# Infraestructura y Basura
ARCHIVE_SUPERSEDED/
temp*/
*_staging*/
_zip_clean*/

# Datos y Resultados pesados
data_intake_*/
results/
BOT_V2_DAYTIME_LAB/outputs/

# Credenciales
mt5_local_config.json
secrets.json
.env
```

## 7. Plan de Sincronización Seguro (Recomendado)

**Camino Elegido: B. Rama de Limpieza**

1. **Crear rama:** `git checkout -b chore/github-clean-sync`
2. **Aplicar .gitignore:** Actualizar el archivo con las nuevas reglas.
3. **Limpiar Cache de Git:** `git rm -r --cached .` (para aplicar el gitignore a archivos ya rastreados).
4. **Add & Commit:** `git add .` (esto solo agregará lo permitido).
5. **Push a Rama:** `git push origin chore/github-clean-sync`
6. **PR:** Revisar en la interfaz de GitHub antes de fusionar a `main`.

## 8. Confirmaciones Operativas
- **Push realizado:** NO.
- **Backtests corridos:** NO.
- **MT5/Real tocado:** NO.

---
**Veredicto Final:** El proyecto está listo para ser versionado, pero requiere la aplicación estricta del `.gitignore` propuesto para evitar el bloqueo de GitHub por archivos grandes o la exposición de llaves de MT5.
