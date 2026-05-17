# TP01 FORMAL DOSSIER: tp01_london_ny_momentum_pullback

## 1. Executive Summary
- **Strategy**: tp01_london_ny_momentum_pullback
- **Rigor**: Train-Only 10-year execution (2015-2024)
- **Timeframe**: M1 (Native resolution)
- **Execution Mode**: normal_mode
- **Cost Profile**: base

## 2. Performance Summary
- **Total Trades**: 191
- **Win Rate**: 47.64%
- **Profit Factor**: 0.90
- **Expectancy (R)**: -0.0684
- **Total Return**: 135.71%
- **Max Drawdown**: 1.32%

## 3. Parameter Settings
{
  "entry_start": "08:00",
  "entry_end": "12:00",
  "ema_period": 20,
  "atr_period": 14,
  "atr_percentile_lookback": 200,
  "atr_percentile": 50.0,
  "momentum_bars": 5,
  "momentum_atr_mult": 1.5,
  "pullback_tolerance_atr": 0.25,
  "stop_atr_buffer": 0.35,
  "target_rr": 2.0,
  "session_name": "all_day"
}

## 4. Yearly Performance Breakdown
| pair | year | trades | wins | losses | breakevens | win_rate | total_pnl_r | total_pnl_usd | avg_pnl_r | max_drawdown_pct | profit_factor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EURUSD | 2015 | 57 | 26 | 31 | 0 | 45.614035087719294 | -11.29303704935716 | -6192.721543033011 | -0.19812345700626596 | 9.486182599751574 | 0.7401780422720786 |
| EURUSD | 2016 | 66 | 32 | 34 | 0 | 48.484848484848484 | -1.8433754399197622 | -2158.3373970820912 | -0.027929930907875183 | 7.501492209771385 | 0.9251023327056904 |
| EURUSD | 2017 | 63 | 31 | 32 | 0 | 49.2063492063492 | 3.137538711977278 | 2602.8087394512186 | 0.049802201777417114 | 6.674192812895825 | 1.0794800603846166 |
| EURUSD | 2018 | 5 | 2 | 3 | 0 | 40.0 | -3.06499445859093 | -3598.056866969111 | -0.612998891718186 | 4.811493963688364 | 0.25271946691699976 |

## 5. Monthly Performance Stats
*Monthly details are saved in tables/TP01_MONTHLY_SUMMARY.csv*
