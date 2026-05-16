# BOT DE TRADING — Laboratorio de Research Cuantitativo

> **Fuente oficial del proyecto:** `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`
> **Runner oficial:** `run_canonical.py`
> **Rama estable Git:** `main`
> **Remote:** `https://github.com/alerasnik97-tech/bottrading.git`

---

## Reglas de trabajo (NO NEGOCIABLES)

| Regla | Descripción |
|---|---|
| **Fuente de verdad** | Solo `C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo`. No trabajar desde copias del escritorio. |
| **Git es la fuente de sincronización** | Antes de pasar trabajo importante a la nube: `git commit && git push`. |
| **El ZIP no es el repo** | `handoff/000_PARA_CHATGPT.zip` es un snapshot de handoff para ChatGPT, no la fuente principal. |
| **No editar en paralelo** | No editar en local y en la nube al mismo tiempo sin push/pull previo. |
| **Backup es solo lectura** | `D:\BACKUPS\BOT DE TRADING` es solo seguridad, nunca fuente activa de trabajo. |

---

## Estructura del proyecto

```
C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\
├── 01_CORE_PRODUCTION          ← Releases aprobadas para producción
├── 02_INCUBATION_STAGING       ← Paper trading y demo controlado
├── 03_RESEARCH_LAB             ← Motor de research, estrategias y backtesting
├── 04_INFRASTRUCTURE_ENGINEERING ← VPS, entorno Python y scripts de soporte
│   └── python_environment/
│       └── requirements.txt
├── 05_MARKET_DATA_VAULT        ← Fuente de verdad de datos (Solo Lectura)
├── 06_GOVERNANCE_AND_COMPLIANCE ← Políticas, auditorías y documentación raíz
│   └── root_docs/
│       └── README.md
├── 07_BACKUPS                  ← Copias de seguridad institucionales
├── 08_CLOUD_FREE_RUN_LAB       ← Laboratorio de ejecución en la nube (Kaggle/Colab)
├── .gitignore
└── .github                     ← Workflows de CI/CD (Excepción técnica)
```

---

## Quick start

**Instalar entorno:**
```bash
pip install -r 04_INFRASTRUCTURE_ENGINEERING/python_environment/requirements.txt
```

**Correr una estrategia (entrypoint canónico):**
```bash
python run_canonical.py <strategy_name> <mode>
# Ejemplo:
python run_canonical.py ny_br_ema normal
```

**Correr tests de infraestructura críticos:**
```bash
python -m pytest research_lab/tests/test_rejection_harness.py research_lab/tests/test_e2e_canonical_flow.py -v
```

---

## Parámetros operativos

| Parámetro | Valor |
|---|---|
| Par | EURUSD |
| Timeframe motor | M15 (datos M5 fuente) |
| Horario operativo | 11:00–19:00 America/New_York |
| Noticias | OFF FORZADO (fail-closed pendiente pipeline UTC) |
| Modo de ejecución | `normal_mode` (default) |
| Capital inicial (simulado) | USD 100.000 |

---

## Estado actual de infraestructura

Ver [`INFRASTRUCTURE_STATUS_FINAL.md`](INFRASTRUCTURE_STATUS_FINAL.md) para el estado sellado de la infraestructura.
Ver [`STRATEGY_PROMOTION_POLICY.md`](STRATEGY_PROMOTION_POLICY.md) para la taxonomía de promoción de estrategias.

---

## Referencias internas

- [`research_lab/README.md`](research_lab/README.md) — Documentación interna del motor
- [`CANONICAL_EXECUTION_CONTRACT.md`](CANONICAL_EXECUTION_CONTRACT.md) — Contrato de ejecución
- [`OOS_REJECTION_PROTOCOL.md`](OOS_REJECTION_PROTOCOL.md) — Protocolo OOS
- [`CLOUD_WORKFLOW.md`](CLOUD_WORKFLOW.md) — Flujo local → Git → Nube
- [`docs/examples/news_example.csv`](docs/examples/news_example.csv) — Ejemplo mínimo de noticias
