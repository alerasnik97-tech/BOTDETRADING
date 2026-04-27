# NEWS IMPACT ANALYSIS REPORT

**Generado:** 2026-04-14 11:33
**Dataset:** official_anchors (190 eventos) vs trades_realistic.csv

---

## RESUMEN EJECUTIVO

- **Total eventos anchor:** 382
- **Total trades analizados:** 910 (2020-2025)
- **Trades por año:** {2022: 99, 2023: 95, 2024: 90, 2025: 113, 2020: 486, 2021: 27}

---

## 1. DISTRIBUCIÓN DE EVENTOS POR GRUPO

### CPI
- Eventos: 83
- Años: [2020, 2021, 2022, 2023, 2024, 2025, 2026]
- Horarios top: 8:00h (83)
- Meses: 83 diferentes

### ECB
- Eventos: 24
- Años: [2024, 2025, 2026]
- Horarios top: 8:00h (15), 7:00h (9)
- Meses: 24 diferentes

### FOMC
- Eventos: 24
- Años: [2024, 2025, 2026]
- Horarios top: 14:00h (24)
- Meses: 24 diferentes

### NFP
- Eventos: 84
- Años: [2020, 2021, 2022, 2023, 2024, 2025, 2026]
- Horarios top: 8:00h (84)
- Meses: 84 diferentes

### PPI
- Eventos: 83
- Años: [2020, 2021, 2022, 2023, 2024, 2025, 2026]
- Horarios top: 8:00h (83)
- Meses: 82 diferentes

### UNEMPLOYMENT
- Eventos: 84
- Años: [2020, 2021, 2022, 2023, 2024, 2025, 2026]
- Horarios top: 8:00h (84)
- Meses: 84 diferentes

## 2. BASELINE DE TRADES (2020-2025)

- **Win Rate global:** 16.0%
- **Avg PnL R:** -0.08R
- **Total trades:** 910

## 3. ANÁLISIS POR VENTANAS DE TIEMPO

Trades que ocurren dentro de X minutos de un evento anchor:

### Ventana ±5 minutos
- **Trades afectados:** 1 (0.1% del total)
- **Win Rate:** 0.0% (Δ -16.0% vs baseline)
- **Avg PnL R:** -1.02R (Δ -0.94R vs baseline)
- **Por anchor group:**
  - PPI: 1 trades, WR=0.0%, PnL=-1.02R

### Ventana ±15 minutos
- **Trades afectados:** 1 (0.1% del total)
- **Win Rate:** 0.0% (Δ -16.0% vs baseline)
- **Avg PnL R:** -1.02R (Δ -0.94R vs baseline)
- **Por anchor group:**
  - PPI: 1 trades, WR=0.0%, PnL=-1.02R

### Ventana ±30 minutos
- **Trades afectados:** 1 (0.1% del total)
- **Win Rate:** 0.0% (Δ -16.0% vs baseline)
- **Avg PnL R:** -1.02R (Δ -0.94R vs baseline)
- **Por anchor group:**
  - PPI: 1 trades, WR=0.0%, PnL=-1.02R

### Ventana ±60 minutos
- **Trades afectados:** 3 (0.3% del total)
- **Win Rate:** 33.3% (Δ +17.3% vs baseline)
- **Avg PnL R:** -0.19R (Δ -0.11R vs baseline)
- **Por anchor group:**
  - NFP: 2 trades, WR=50.0%, PnL=0.22R
  - PPI: 1 trades, WR=0.0%, PnL=-1.02R

## 4. TRADES MÁS CERCANOS A EVENTOS

(Top 20 trades con menor distancia a cualquier evento anchor)

| Min | Grupo | Result | PnL R | Entry Time |
|-----|-------|--------|-------|------------|
| 0 | PPI | loss | -1.02 | 2022-03-15T08:30 |
| 45 | NFP | win | 1.48 | 2025-03-07T07:45 |
| 60 | NFP | loss | -1.04 | 2022-09-02T07:30 |
| 75 | PPI | win | 1.47 | 2022-06-14T09:45 |
| 75 | NFP | loss | -1.03 | 2024-07-05T09:45 |
| 75 | PPI | win | 1.46 | 2024-08-13T07:15 |
| 75 | FOMC | win | 1.47 | 2025-06-18T12:45 |
| 90 | PPI | win | 1.46 | 2023-12-13T07:00 |
| 90 | PPI | win | 1.48 | 2024-04-11T10:00 |
| 90 | NFP | win | 1.46 | 2024-05-03T07:00 |
| 105 | CPI | loss | -1.01 | 2023-02-14T10:15 |
| 105 | FOMC | loss | -1.07 | 2024-05-01T12:15 |
| 105 | FOMC | loss | -1.02 | 2024-11-07T12:15 |
| 120 | PPI | loss | -1.03 | 2023-11-15T06:30 |
| 150 | CPI | loss | -0.63 | 2023-05-10T11:00 |
| 150 | NFP | loss | -0.22 | 2024-05-03T11:00 |
| 150 | PPI | loss | -1.03 | 2025-01-14T06:00 |
| 150 | CPI | loss | 0.00 | 2020-04-10T11:00 |
| 150 | CPI | loss | 0.00 | 2020-10-13T11:00 |
| 155 | NFP | loss | 0.00 | 2020-05-01T11:05 |