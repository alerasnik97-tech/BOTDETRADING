# PHASE28: WINRATE + FREQUENCY IMPROVEMENT STUDY

- **Timestamp:** 2026-04-28T19:33:00-03:00
- **Veredicto:** PHASE28_BALANCED_IMPROVEMENT_FOUND
- **Phase25 sigue autoridad:** SÍ (hasta auditoría forense del candidato)

## Baseline Phase25
| Métrica | Valor |
|---|---|
| Sample | 2,625 |
| PF | 2.79 |
| Exp | 0.281R |
| WR | 32.5% |
| Max DD | -5.58R |
| Max Streak | 14 |
| TPM | 19.4 |
| Meses <15 | 0 |

## Hallazgos Clave

### ¿WR 50% es viable?
**SÍ, pero destruye el edge.** NO_BE alcanza WR 58.6% pero DD sube a -10.57R. TP1.2_NOBE alcanza 62.2% pero DD = -8.84R. **Rechazado.**

### ¿15 trades todos los meses?
**YA SE CUMPLE.** La baseline Phase25 tiene 0 meses con <15 trades.

### Mejor candidato WR con PF aceptable
**BE_0.6:** WR=40.0%, PF=2.55, Exp=0.323R, DD=-7.64R. Rechazado por DD excesivo.

**BE_0.5:** WR=36.5%, PF=2.64, Exp=0.304R, DD=-5.6R, Streak=12. Interesante pero DD similar.

### Mejor candidato balanceado
**TP1.2_BF65:** WR=36.3%, PF=2.75, Exp=0.271R, DD=-5.1R, Streak=15, TPM=19.8
- Walk-forward: **5/5 passes**
- WR sube +3.8pp vs baseline
- PF baja marginalmente (-0.04)
- DD **mejora** (-5.1 vs -5.58)

## Hipótesis Individuales (Top)
| Hipótesis | WR | PF | Exp | DD | Streak | TPM |
|---|---|---|---|---|---|---|
| BASELINE | 32.5 | 2.79 | 0.281 | -5.58 | 14 | 19.4 |
| TP_1.2 | **36.6** | 2.72 | 0.270 | -5.18 | **12** | 19.4 |
| TP_1.3 | 34.5 | 2.77 | 0.277 | -5.78 | **12** | 19.4 |
| BE_0.5 | **36.5** | 2.64 | **0.304** | -5.6 | **12** | 19.4 |
| NO_BE | **58.6** | 1.98 | **0.376** | -10.57 | **8** | 19.4 |
| WIN_07_13 | 32.8 | **2.88** | 0.290 | -5.5 | 13 | 18.3 |
| 2TRADES | 32.0 | 2.64 | 0.263 | -6.39 | 17 | **32.8** |

## Combinaciones (Top)
| Combo | WR | PF | Exp | DD | Streak | TPM |
|---|---|---|---|---|---|---|
| TP1.2_BF65 | **36.3** | 2.75 | 0.271 | **-5.1** | 15 | 19.8 |
| TP1.3_BF65 | 34.3 | 2.81 | 0.280 | -5.7 | 15 | 19.8 |
| TP1.2_BF60 | 35.9 | 2.71 | 0.266 | -5.48 | 15 | 20.3 |
| TP1.2_NOBE | **62.2** | 2.02 | 0.356 | -8.84 | 8 | 19.4 |

## Walk-Forward
| Candidato | Passes | Total |
|---|---|---|
| TP1.2_BF65 | **5/5** | 100% |
| TP1.2_W0813 | **5/5** | 100% |
| TP1.2_BF60_2T | **5/5** | 100% |

## Conclusión
Se encontró un candidato balanceado (**TP1.2_BF65**) que sube WR de 32.5% a 36.3%, reduce DD de -5.58R a -5.1R, y pasa walk-forward 5/5. PF baja marginalmente (2.75 vs 2.79). Phase25 sigue como autoridad hasta auditoría forense del candidato.

## Siguiente Paso Único
Auditoría forense del candidato TP1.2_BF65 antes de considerar reemplazo de Phase25.
