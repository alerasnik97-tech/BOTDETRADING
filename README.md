# BOT DE TRADING — Laboratorio de Research Cuantitativo

> **Fuente oficial del proyecto:** `C:\BOT DE TRADING`
> **Runner oficial:** `run_canonical.py`
> **Rama estable Git:** `main`
> **Remote:** `https://github.com/alerasnik97-tech/bottrading.git`

---

## Reglas de trabajo (NO NEGOCIABLES)

| Regla | Descripción |
|---|---|
| **Fuente de verdad** | Solo `C:\BOT DE TRADING`. No trabajar desde copias del escritorio. |
| **Git es la fuente de sincronización** | Antes de pasar trabajo importante a la nube: `git commit && git push`. |
| **El ZIP no es el repo** | `handoff/000_PARA_CHATGPT.zip` es un snapshot de handoff para ChatGPT, no la fuente principal. |
| **No editar en paralelo** | No editar en local y en la nube al mismo tiempo sin push/pull previo. |
| **Backup es solo lectura** | `D:\BACKUPS\BOT DE TRADING` es solo seguridad, nunca fuente activa de trabajo. |

---

## Estructura del proyecto

```
C:\BOT DE TRADING\
├── research_lab/           ← Motor oficial, estrategias, validación, tests
│   ├── main.py             ← Orquestador canónico
│   ├── engine.py           ← Motor de backtesting
│   ├── validation.py       ← WFA y harness OOS
│   ├── rejection_protocol.py ← Protocolo de rechazo IS/OOS
│   ├── strategies/         ← Estrategias activas
│   ├── tests/              ← Tests críticos de infraestructura
│   └── version.py          ← Versionado del laboratorio
├── run_canonical.py        ← Único entrypoint autorizado
├── data_free_2020/prepared/EURUSD_M5.csv    ← Dataset IS 2020-2021
├── data_candidates_2022_2025/prepared/EURUSD_M5.csv ← Dataset OOS 2022-2025
├── handoff/                ← ZIP de intercambio ChatGPT (no fuente)
├── docs/                   ← Documentación técnica
├── legacy/                 ← Código archivado (no activo)
├── requirements.txt
├── STRATEGY_PROMOTION_POLICY.md
├── OOS_REJECTION_PROTOCOL.md
├── CANONICAL_EXECUTION_CONTRACT.md
├── COMPARABILITY_2020_2025_NOTE.md
├── INFRASTRUCTURE_STATUS_FINAL.md
└── CLOUD_WORKFLOW.md
```

---

## Quick start

**Instalar entorno:**
```bash
pip install -r requirements.txt
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
