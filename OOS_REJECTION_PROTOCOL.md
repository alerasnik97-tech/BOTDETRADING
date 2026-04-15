# PROTOCOLO DE RECHAZO CUANTITATIVO OOS (OOS Rejection Protocol)

## 1. FILOSOFÍA CENTRAL
El objetivo del laboratorio F1 de `BOT DE TRADING CURSOR` ya NO es "encontrar" estrategias ganadoras por fuerza bruta o data-mining estético. Su objetivo es **DESTROZAR iterativamente el ruido**. Sólo las estrategias que sobrevivan robustamente a los costos operacionales estrictos (penalidad rollover, spread, slippage gap) merecen ancho de banda mental.
Prevenir el reporte de basura cuantitativa (overfitting in-sample sin edge) ahorra CPU, disco y energía psicológica.

## 2. NIVELES DE CLASIFICACIÓN
El evaluador emite una de cuatro señales tras procesar una grilla de parámetros:

1. **`hard_reject`**: Aborto absoluto. La estrategia es un suicidio matemático o carece de muestra (e.g. Pf_OOS < 1.0, o se ejecuta <1 vez al mes).
2. **`soft_reject`**: El edge puede sumar positivo (Expectancy marginal), pero la ejecución es aberrante (e.g. Drawdown insano >15% sobre 0.5% risk base o alta inconsistencia anual).
3. **`pass_minimum`**: Supera el filtro de vida (PF_OOS > 1.01, DD aceptable). Es material para estudiar o refinar en variaciones de mercado, pero no apto para live.
4. **`strong_candidate`**: Aprobación dorada (PF > 1.20, Exp > 0.10, Drawdown < 8.0%, <= 1 año negativo). Estrategias listas para incubación extrema.

## 3. SHORT-CIRCUIT IN-SAMPLE (EARLY ABORT)
**Regla:** Si la optimización exhaustiva In-Sample (IS) —en el terreno que la estrategia *debería dominar*— no es capaz de mostrar un resultado estelar absoluto (PF mínimo 1.05 y Expectancy 0.02R), la estrategia denota carencia total de edge. 
**Acción:** El sistema **aborta inmediatamente**, esquiva ejecutar Walk Forward Analysis (WFA) ahorrando 80% de recursos, no dibuja curvas de Equity y graba directo en la carpeta `REJECTION_REPORT.md` (Fase: `IN_SAMPLE_FLOP`).

## 4. FATAL THRESHOLDS OOS (WFA PENALTIES)
Si logró sobrevivir el filtro base IS, se somete a sus datos ciegos (OOS). La estrategia sufrirá `hard_reject` si en la agregación OOS WFA comete fallos irredimibles:
- Expectancy_R o Profit Factor neto negativo/1.0. (Significa que pierde al sumarle los castigos de late-session).
- Falta de muestra estadística penalizada que inhabilita confianza (`insufficient_sample == True`).
Y sufrirá `soft_reject` en OOS si:
- Drawdown Maximum excedió el violento threshold del 15% (OOSS_DRAWDOWN_UNACCEPTABLE).
- Mantuvo consistencia desastrosa perdiendo de base 3 años naturales distintos.

## 5. TRAZABILIDAD
No existe más el "bot no sirvió". Ahora el laboratorio arroja una etiqueta innegable por estrategia en la matriz terminal y lo escribe explícitamente en:
`results/research_lab_robust/[estrategia]/REJECTION_REPORT.md`
En él se explícita la razón determinística (`is_reason`, `oos_reason`) sin maquillar el fracaso.
