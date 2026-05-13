# MANIPULANTE 4 — M3 RED HANDOFF

## 1. Por qué M3.0 quedó RED

MANIPULANTE 3.0 fue declarado RED tras auditoría externa completa porque:

- **30/30 cells FTMO blown.** Todas las configs, todas las fases, todos los slippage. 100% catástrofe.
- **PF máximo en VAL = 0.8918** (CFG_004, slippage 0.2). Pierde 1.12R por cada 1R ganado.
- **PF máximo en TEST = 0.8375** (CFG_001). Ninguna config supera 1.0 en ninguna fase.
- **CFG_004 "mejor" es engañosa:** PF_TRAIN=0.29, PF_VAL=0.89, PF_TEST=0.30. El pico en VAL es ruido sobre muestra perdedora.
- **DD destructor:** hasta -23.18R en TEST, -19.37R en VAL.
- **EOM blocker corregido limpiamente:** 0 artificial EOM en métricas, independent verify match exacto. El blocker no causó el RED.

## 2. Qué NO debe repetirse

- **No tratar todos los barridos como equivalentes.** M3 no discriminaba calidad del sweep.
- **No usar CHOCH genérico sin intención.** M3 aceptaba cualquier cambio estructural LTF sin verificar displacement real.
- **No correr 5 configs extremas y extrapolar.** La pilot original tuvo solo 5 configs que cubrían pocas combinaciones reales.
- **No ignorar que TRAIN pierde.** CFG_004 con PF_TRAIN=0.29 no debió pasar a VAL/TEST.
- **No label INCONCLUSIVE cuando la evidencia es RED claro.** PF<1.0 uniforme con N suficiente = RED.

## 3. Qué parte de la lógica madre todavía puede investigarse

Per la auditoría externa (MANIPULANTE3_EXTERNAL_AUDIT.md):

> "La familia conceptual 'barrido de calidad + displacement' merece exactamente una oportunidad más con una traducción fundamentalmente diferente — un micro-probe con discriminación objetiva de calidad del barrido, muerte rápida honesta, y cero cosmética paramétrica. Si ese probe también falla, la rama entera se cierra."

Lo que queda vivo: la discriminación programable de calidad del barrido + evidencia de intención post-sweep.

## 4. Oportunidad legítima: sweep quality + displacement gate

La auditoría externa identificó 3 variables críticas:

1. **Sweep distance / ATR ratio:** ¿Fue barrido real o tick noise?
2. **Displacement magnitude post-sweep:** ¿Cuerpo de rechazo > X% del ATR? Distingue rechazo real de asomo tímido.
3. **Close back inside previous range:** Confirma intención de reversa, no solo wick.

Variables prohibidas por la auditoría:
- Sweep speed en milisegundos (overfitting a microestructura).
- Proximidad continua a niveles (data mining).
- Filtros de volatilidad paramétricos (demasiado parámetricos).
- Contexto post-news como variable continua.

## 5. Riesgos metodológicos que arrastra M4

- **Herencia de motor:** El motor V7 es el mismo. Si tiene bugs ocultos, M4 hereda la falla.
- **Herencia de datos:** Misma data tick. Si hay gaps no detectados, M4 los hereda.
- **Riesgo de cosmética:** Agregar 2-3 filtros sobre una base perdedora puede generar "edge" por selección estadística.
- **Riesgo emocional:** Después de 3 iteraciones RED, hay presión para encontrar resultado positivo.

## 6. Criterios de muerte rápida M4

- TRAIN PF_net < 1.0 → muerte inmediata sin mirar VAL.
- VAL PF_net < 1.10 con N>=40 bajo slippage 0.2 → muerte inmediata.
- FTMO blown en mayoría de configs → muerte.
- EOM artificial > 0 en métricas → blocker.
- Slippage 0.2 destruye el edge completamente → muerte.
- Profit concentration extrema (>60% en <=3 trades) → muerte.

## Fuentes leídas

- MANIPULANTE3_RED_FINAL_DECISION.md
- MANIPULANTE3_EXTERNAL_AUDIT.md
- MANIPULANTE3_EDGE_TRANSLATION_DIAGNOSIS.md
- NEXT_EDGE_HYPOTHESIS_RECOMMENDATION.md
- MAX_CONFIRMATION_EOM_FIXED_RESULTS_VAL.csv
- MAX_CONFIRMATION_EOM_FIXED_RESULTS_TEST.csv
