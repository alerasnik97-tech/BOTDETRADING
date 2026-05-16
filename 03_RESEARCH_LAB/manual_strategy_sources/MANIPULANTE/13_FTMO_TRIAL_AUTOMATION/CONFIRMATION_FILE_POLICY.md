# Confirmation File Policy

Required file:

`MANIPULANTE\13_FTMO_TRIAL_AUTOMATION\I_CONFIRM_FTMO_TRIAL_AUTO.txt`

Exact content:

```text
I UNDERSTAND THIS IS FTMO FREE TRIAL DEMO ONLY
I CONFIRM NO REAL MONEY
I CONFIRM MANIPULANTE ONLY
RISK_DEFAULT=0.50
ONE_TRADE_PER_DAY
NEWS_GATE_REQUIRED
DATA_GATE_REQUIRED
```

Do not create this file while blockers exist. Required gates:

- FTMO demo/trial account confirmed.
- News Gate = ALLOW.
- Week news loaded = true.
- Data/Time/Symbol/Lot Gates = ALLOW.
- Signal Sync = OK.
- Dry-run = PASS.
- STOP_BOT removed intentionally.
