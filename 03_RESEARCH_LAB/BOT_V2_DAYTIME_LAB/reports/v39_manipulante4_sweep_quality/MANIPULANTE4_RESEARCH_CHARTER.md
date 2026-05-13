# MANIPULANTE 4 — RESEARCH CHARTER

## Identity
- **Name:** MANIPULANTE 4 — Sweep Quality + Displacement Gate
- **Family:** Barrido de liquidez + cambio de estructura
- **Predecessor:** MANIPULANTE 3.0 (RED, sealed)
- **Nature:** Nueva traducción programable, NO continuación optimizada de M3

## Mandate

1. M4 NO es optimización cosmética sobre M3. Es una traducción fundamentalmente diferente que mide CALIDAD del barrido e INTENCIÓN del desplazamiento posterior.
2. Se prohíbe fuerza bruta. Máximo 150 configs en micro-probe.
3. Se prohíbe optimización emocional o post-hoc.
4. La prioridad es medir si "calidad del barrido + desplazamiento real" aporta edge medible.
5. La data manual del usuario es referencia conceptual, no target a replicar.
6. TEST no se usa para seleccionar ni rankear. Solo verificación pass/fail final.
7. Métrica oficial = net_r (post comisión + slippage).
8. Slippage 0.2 pips obligatorio para aprobación.
9. Data/News constraints heredadas de M3: fail-close, Tier-1 buffers, rollover block.
10. EOM artificial en métricas = blocker inmediato.
11. Forward demo futuro obligatorio si alguna vez hay candidata viable.

## Success Metrics (pre-committed, immutable)
- PF_val_net (slip 0.2) >= 1.15
- PF_test_net (slip 0.2) >= 1.00
- N_val >= 40
- N_test >= 40
- FTMO not blown tempranamente
- EOM artificial in metrics = 0
- Slippage 0.2 viable (PF no colapsa vs 0.0)
- Independent verify match

## Kill Criteria (pre-committed, immutable)
- TRAIN PF_net < 1.0 → muerte inmediata
- VAL PF_net (slip 0.2) < 1.05 → muerte
- FTMO blown en mayoría de configs → muerte
- EOM artificial > 0 en métricas → blocker
- Profit concentration > 60% en ≤3 trades → muerte
- N_val < 40 → inconcluso, no escalar

## What M4 measures that M3 did NOT
- Sweep depth relative to ATR (quality gate, not binary existence)
- Reclaim speed (close back inside level in constrained time window)
- Displacement body magnitude (ATR-normalized, not just any CHOCH)
- Structure break AFTER displacement (not before)
- Entry AFTER observable intention (not at first weak micro-choch)
