# PHASE 26-B: DATA CERTIFICATION CHECKLIST 2015-2019

- [ ] Timestamps parseables y sin saltos temporales ilógicos.
- [ ] Timezone validada (UTC o EST explícito).
- [ ] Sin duplicados.
- [ ] Identificación y métrica de Gaps.
- [ ] BID <= ASK estrictamente (sin spreads negativos).
- [ ] Spread positivo y spread extremo auditado.
- [ ] OHLC válido (High >= Low, High >= Open/Close).
- [ ] Continuidad garantizada por año.
- [ ] Generación exitosa de M3 a partir de M1/Tick.
- [ ] Generación de Data Quality Mask 2015-2019.
- [ ] Generación y alineación de News Fortress 2015-2019.
- [ ] Clasificación final: CERTIFIED_WITH_MASK (ideal).
