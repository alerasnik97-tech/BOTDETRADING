# PHASE50T TIME_EXIT RULE SOURCE AUDIT

## 1. Engine Findings
- The engine code (`research_lab/engine.py`) explicitly implements `time_exit` logic.
- It is triggered when `held_bars >= max_hold_bars`.
- Another trigger is `forced_session_close` based on `force_close_minute`.

## 2. Strategy Findings
- `MANIPULANTE_STRATEGY_CARD.md` does NOT list `max_hold_bars` as a core parameter.
- It ONLY mentions `Global Weekend Hard Close` at Friday 16:55 NY.
- The `manipulante_config.json` also excludes `max_hold_bars`.

## 3. Preliminary Conclusion
- `TIME_EXIT` appears to be a generic engine safety rule that was active during the generation of the `phase38` dataset, even if not explicitly defined as a MANIPULANTE core rule.
- Its high prevalence in August 2017 suggests the engine was 'killing' trades at a session end or bar limit.
