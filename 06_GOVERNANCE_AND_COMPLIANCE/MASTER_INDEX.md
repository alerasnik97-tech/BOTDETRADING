# MASTER INDEX — BOT DE TRADING V7

## Estado Actual
- Motor UnifiedV7Engine validado, CostModel integrado, full suite 207/207
- Gate 6FB2 PASS, Gate 6F Claude-proof ready
- Arquitectura institucional 7 carpetas migrada (Gate 7)
- Multi-agent control plane activo (Gate 8)
- Laboratorio real NO ejecutado aun

## Arquitectura
| Carpeta | Proposito | Owner |
|---|---|---|
| 01_CORE_PRODUCTION | Produccion congelada | Production Release Agent |
| 02_INCUBATION_STAGING | Forward/paper testing | Incubation Agent |
| 03_RESEARCH_LAB | Backtests, motor V7, gates | Research Agent |
| 04_INFRASTRUCTURE | Packaging, monitoring | Infrastructure Agent |
| 05_MARKET_DATA_VAULT | Datos READ-ONLY | Data Quality Agent |
| 06_GOVERNANCE | Reglas, protocolos | Governance Agent |
| 07_BACKUPS | Cold storage | Backup Agent |

## ZIP Oficial: `000_PARA_CHATGPT.zip` (unico, en raiz)
## Tests: `python -m pytest -o pythonpath=. src/v7_engine/tests/ -v` desde 03_RESEARCH_LAB/BOT_V2_DAYTIME_LAB
## Proximo paso: reintentar Gate 6 controlado dentro de la arquitectura multi-agente
