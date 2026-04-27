# Next Experiment Design

## Hypothesis

Si el edge residual de EURUSD PM existe, no esta en una lectura M15 amplia sino en micro-exhaustions intradia que se extienden demasiado contra VWAP y luego recuperan rapido. Esa hipotesis exige menor tiempo expuesto y ejecucion de precision.

## Architecture

- Context: H1 para regimen.
- Signal frame: M3.
- Execution frame: M1 BID/ASK real.
- Session: 11:30-16:00 NY.
- Max trades: 1 por dia.

## Entry Logic

- Solo dentro de PM.
- Se exige stretch contra VWAP en desviacion estandar.
- Se exige sweep de extremo local reciente.
- Se exige reclaim del cuerpo de la vela senal.
- Se exige filtro de ADX H1 para evitar perseguir tendencia fuerte.

## Exit Logic

- Stop hard estructural desde el inicio, usando el extremo de la vela senal mas buffer ATR.
- Target fijo en multiples R.
- Time-exit exacto de pocos bares.
- Forced flat por horario.
- News Fortress activo sobre dataset PM-safe.

## Risk Controls

- Stop hard obligatorio y auditado por engine.
- Rechazo de senales sin hard stop.
- Kill de posicion y kill de pending signal en ventana de noticias bloqueadas.
- Dataset PM-safe exact-time con pre-block de 45m y cooldown/post-block de 90m.

## Invalidating Conditions

La idea queda invalida si:

- sigue generando muestra demasiado chica,
- o mejora solo por 1-2 trades,
- o el holdout no confirma,
- o la robustez depende de thresholds muy finos.
