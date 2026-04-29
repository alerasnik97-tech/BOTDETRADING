# PHASE26-C: FULL 2015-2019 DATA CERTIFICATION REPORT

- **Timestamp:** 2026-04-28T18:38:00-03:00
- **Veredicto:** PHASE26C_2015_2019_DATA_CERTIFIED_WITH_MASK
- **Phase25:** Sigue siendo autoridad. Congelada. No se tocó.
- **Optimización:** Sigue bloqueada hasta validar Phase25 en 2015-2026.

## Fuente
- Local Dukascopy M1 BID/ASK real.
- Ubicación: `data_intake_2015_2019/raw_m1/{año}/`
- ~23 MB por archivo por año.

## M1 Normalización Full
| Año  | Filas   | Gaps | Dupes | Neg Spread | Veredicto               |
|------|---------|------|-------|------------|-------------------------|
| 2015 | 372,928 | 527  | 0     | 0          | M1_CERTIFIED_WITH_WARNINGS |
| 2016 | 373,767 | 421  | 0     | 3          | M1_CERTIFIED_WITH_WARNINGS |
| 2017 | 373,041 | 348  | 0     | 0          | M1_CERTIFIED_WITH_WARNINGS |
| 2018 | 373,448 | 145  | 0     | 0          | M1_CERTIFIED_WITH_WARNINGS |
| 2019 | 373,253 | 257  | 0     | 0          | M1_CERTIFIED_WITH_WARNINGS |

Nota: Los gaps son exclusivamente fines de semana y baja liquidez. Cero duplicados. Solo 3 spreads negativos marginales en 2016 (bloqueados por mask).

## M3 Generación desde M1
| Año  | Filas M3 | Gaps M3 | Neg Spread | Veredicto             |
|------|----------|---------|------------|-----------------------|
| 2015 | 124,501  | 66      | 0          | M3_CERTIFIED_WITH_MASK |
| 2016 | 124,753  | 78      | 1          | M3_CERTIFIED_WITH_MASK |
| 2017 | 124,465  | 66      | 0          | M3_CERTIFIED_WITH_MASK |
| 2018 | 124,517  | 56      | 0          | M3_CERTIFIED_WITH_MASK |
| 2019 | 124,495  | 57      | 0          | M3_CERTIFIED_WITH_MASK |

Total M3: 622,731 barras.

## Data Quality Mask
- Total barras: 622,731
- Bloqueadas: 1 (0.00%)
- Fail-closed: SÍ
- Veredicto: MASK_FULL_CREATED

## News Fortress
- Total eventos: 380
- Años cubiertos: 2015-2019
- Veredicto: NEWS_CERTIFIED

## Certificación por Año
| Año  | Veredicto           |
|------|---------------------|
| 2015 | CERTIFIED_WITH_MASK |
| 2016 | CERTIFIED_WITH_MASK |
| 2017 | CERTIFIED_WITH_MASK |
| 2018 | CERTIFIED_WITH_MASK |
| 2019 | CERTIFIED_WITH_MASK |

## Conclusión
La data 2015-2019 está certificada con mask. Phase25 puede validarse sobre 2015-2026 completo como siguiente paso. NO se optimiza todavía.

## Siguiente Paso Único
Validar Phase25 exactamente igual en el rango 2015-2026 completo.
