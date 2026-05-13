# REGISTRO DE AGENTES ACTIVOS (ACTIVE AGENT REGISTRY)

## Agente 1 — Research
- **nombre lógico:** Research Agent
- **tarea:** MANIPULANTE 3.0 HTF/LTF Research
- **carpeta permitida de escritura:**
  `03_RESEARCH_LAB\`
- **carpetas permitidas de lectura:**
  `05_MARKET_DATA_VAULT\`
  `06_GOVERNANCE_AND_COMPLIANCE\`
- **prohibiciones:**
  no producción, no incubación, no backups, no push, no Explorer
- **riesgo principal:**
  contaminar research con TEST, tocar datos o correr sweep sin pilot

## Agente 2 — Data/News
- **nombre lógico:** Data Quality Agent
- **tarea:** Data + News Quality Audit READ-ONLY
- **escritura permitida:**
  `06_GOVERNANCE_AND_COMPLIANCE\data_quality_audits\`
- **lectura permitida:**
  `05_MARKET_DATA_VAULT\`
  `06_GOVERNANCE_AND_COMPLIANCE\`
- **prohibiciones:**
  no modificar datos, no tocar research, no runner, no strategy
- **riesgo principal:**
  tocar datos en vez de solo auditar

## Agente 3 — Governance Control Board
- **nombre lógico:** Governance Supervisor
- **tarea:** coordinación y control
- **escritura permitida:**
  `06_GOVERNANCE_AND_COMPLIANCE\multi_agent_control\`
- **prohibiciones:**
  no código, no datos, no runner, no tests, no ZIP, no backtests
- **riesgo principal:**
  sobreintervenir o bloquear sin evidencia
