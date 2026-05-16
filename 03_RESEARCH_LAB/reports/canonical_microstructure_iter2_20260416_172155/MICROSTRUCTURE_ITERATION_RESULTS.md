# Microstructure Iteration Results

## Runs Compared

- benchmark diagnostico anterior: `results/pm_micro_reclaim_m3/20260416_145640_pm_micro_reclaim_m3`
- iteracion seria actual: `results/pm_micro_reclaim_m3/20260416_161425_pm_micro_reclaim_m3`

## Benchmark vs Iteration

### Muestra

- old full trades: 5
- new full trades: 37
- old avg trades/month: 0.0694
- new avg trades/month: 0.5139

Lectura:

- la iteracion si mejoro la muestra de forma fuerte
- la pregunta correcta era si esa muestra adicional preservaba calidad

### Calidad y robustez

Full sample:

- old PF: 1.8649
- new PF: 0.2611

- old expectancy R: 0.2852
- new expectancy R: -0.4826

- old max DD %: 0.7071
- new max DD %: 10.6015

- old negative years: 2
- new negative years: 6

Segmentos:

- development: 2 trades / PF 2.162 -> 22 trades / PF 0.150
- validation: 2 trades / PF inf -> 6 trades / PF 0.210
- holdout: 1 trade / PF 0.000 -> 9 trades / PF 0.742

Lectura:

- la muestra subio
- la robustez no mejoro
- cuando la logica obtiene masa critica, el edge colapsa

### Resultado operacional

En la iteracion seleccionada:

- 37 trades
- win rate 21.62%
- 25 stop losses
- 10 time exits
- 2 take profits
- 0 news exits

Esto no describe una ineficiencia microestructural sana. Describe una hipotesis que, al relajarse lo suficiente para generar muestra, deja de sostener la calidad economica.

## Serious Gate Outcome

Ninguna combinacion paso el serious gate definido antes de correr.

## Professional Verdict

El aumento de muestra fue real pero inutil desde el punto de vista de edge.

Conclusion final:

`Close the line`

No corresponde seguir iterando `pm_micro_reclaim_m3` como hipotesis viva.
