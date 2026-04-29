# PHASE32B FUNDEDNEXT STELLAR LITE 10K SIMULATION REPORT

## Objetivo
Simulate whether Phase25 authority supports FundedNext Stellar Lite 10k, with special focus on 0.75% risk.

## Estrategia simulada
- PHASE25_AUTHORITY only.
- TP1.4 / BE0.4 / BF70.
- No shadow candidate.

## Reglas Stellar Lite 10k
- Phase 1 target: 8% / 800 USD.
- Phase 2 target: 4% / 400 USD.
- Daily loss: 4% / 400 USD.
- Max loss: 8% / 800 USD.
- Minimum trading days/trades: 5 per phase.
- Time limit: unlimited.

## Resultados clave
- 0.50%: combined pass 97.79%, daily breach 0.0%, max breach 0.0%.
- 0.75%: combined pass 93.38%, daily breach 5.15%, max breach 0.0%.
- 1.00%: combined pass 91.91%, daily breach 6.62%, max breach 0.0%.

## Daily loss 4%
{
  "daily_loss_limit_pct": 4.0,
  "intraday_equity_mode": "mae_proxy",
  "breach_cases": 24,
  "breaches_by_risk": {
    "0.75": 2,
    "0.85": 2,
    "1.0": 4,
    "1.25": 8,
    "1.5": 8
  },
  "risk_075_supports_daily_loss_4pct": false,
  "risk_050_more_prudent": true,
  "risk_100_discarded_as_base": true,
  "pure_sl_streak_4_implication": "At 0.75% four pure SL equals 3.0%, below 4% daily loss but close after MAE; at 1.0% four SL equals the full 4% limit.",
  "most_dangerous_dates": [
    {
      "strategy": "PHASE25",
      "risk_pct": 1.5,
      "trade_id": "PHASE25_02320",
      "entry_time": "2025-01-20 07:27:00-05:00",
      "balance_start_day_pct": 1010.1266,
      "equity_low_estimate_pct": 1001.1386,
      "closed_pnl_pct": -1.5,
      "open_pnl_proxy_pct": -8.988,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -4.988,
      "mae_r": -5.992
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.5,
      "trade_id": "PHASE25_00715",
      "entry_time": "2018-02-14 07:36:00-05:00",
      "balance_start_day_pct": 320.1309,
      "equity_low_estimate_pct": 311.5825,
      "closed_pnl_pct": -1.5,
      "open_pnl_proxy_pct": -8.5484,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -4.5484,
      "mae_r": -5.6989
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.25,
      "trade_id": "PHASE25_02320",
      "entry_time": "2025-01-20 07:27:00-05:00",
      "balance_start_day_pct": 841.7721,
      "equity_low_estimate_pct": 834.2821,
      "closed_pnl_pct": -1.25,
      "open_pnl_proxy_pct": -7.49,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -3.49,
      "mae_r": -5.992
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.25,
      "trade_id": "PHASE25_00715",
      "entry_time": "2018-02-14 07:36:00-05:00",
      "balance_start_day_pct": 266.7757,
      "equity_low_estimate_pct": 259.6521,
      "closed_pnl_pct": -1.25,
      "open_pnl_proxy_pct": -7.1237,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -3.1237,
      "mae_r": -5.6989
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.5,
      "trade_id": "PHASE25_02461",
      "entry_time": "2025-08-22 09:36:00-04:00",
      "balance_start_day_pct": 1055.9957,
      "equity_low_estimate_pct": 1049.6327,
      "closed_pnl_pct": 0.0,
      "open_pnl_proxy_pct": -6.3631,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -2.3631,
      "mae_r": -4.242
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.5,
      "trade_id": "PHASE25_00602",
      "entry_time": "2017-08-11 07:57:00-04:00",
      "balance_start_day_pct": 268.1299,
      "equity_low_estimate_pct": 261.9871,
      "closed_pnl_pct": -1.5,
      "open_pnl_proxy_pct": -6.1429,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -2.1429,
      "mae_r": -4.0952
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.0,
      "trade_id": "PHASE25_02320",
      "entry_time": "2025-01-20 07:27:00-05:00",
      "balance_start_day_pct": 673.4177,
      "equity_low_estimate_pct": 667.4257,
      "closed_pnl_pct": -1.0,
      "open_pnl_proxy_pct": -5.992,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -1.992,
      "mae_r": -5.992
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.0,
      "trade_id": "PHASE25_00715",
      "entry_time": "2018-02-14 07:36:00-05:00",
      "balance_start_day_pct": 213.4206,
      "equity_low_estimate_pct": 207.7217,
      "closed_pnl_pct": -1.0,
      "open_pnl_proxy_pct": -5.6989,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -1.6989,
      "mae_r": -5.6989
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.5,
      "trade_id": "PHASE25_00546",
      "entry_time": "2017-05-12 07:45:00-04:00",
      "balance_start_day_pct": 244.8996,
      "equity_low_estimate_pct": 239.4996,
      "closed_pnl_pct": -1.5,
      "open_pnl_proxy_pct": -5.4,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -1.4,
      "mae_r": -3.6
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.25,
      "trade_id": "PHASE25_02461",
      "entry_time": "2025-08-22 09:36:00-04:00",
      "balance_start_day_pct": 879.9964,
      "equity_low_estimate_pct": 874.6939,
      "closed_pnl_pct": 0.0,
      "open_pnl_proxy_pct": -5.3025,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -1.3025,
      "mae_r": -4.242
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.25,
      "trade_id": "PHASE25_00602",
      "entry_time": "2017-08-11 07:57:00-04:00",
      "balance_start_day_pct": 223.4416,
      "equity_low_estimate_pct": 218.3225,
      "closed_pnl_pct": -1.25,
      "open_pnl_proxy_pct": -5.119,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -1.119,
      "mae_r": -4.0952
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 0.85,
      "trade_id": "PHASE25_02320",
      "entry_time": "2025-01-20 07:27:00-05:00",
      "balance_start_day_pct": 572.4051,
      "equity_low_estimate_pct": 567.3119,
      "closed_pnl_pct": -0.85,
      "open_pnl_proxy_pct": -5.0932,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -1.0932,
      "mae_r": -5.992
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.5,
      "trade_id": "PHASE25_01305",
      "entry_time": "2020-09-16 13:24:00-04:00",
      "balance_start_day_pct": 560.1753,
      "equity_low_estimate_pct": 555.1483,
      "closed_pnl_pct": 0.0,
      "open_pnl_proxy_pct": -5.027,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -1.027,
      "mae_r": -3.3514
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.5,
      "trade_id": "PHASE25_00469",
      "entry_time": "2017-01-06 07:57:00-05:00",
      "balance_start_day_pct": 217.9768,
      "equity_low_estimate_pct": 212.9768,
      "closed_pnl_pct": 0.0,
      "open_pnl_proxy_pct": -5.0,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -1.0,
      "mae_r": -3.3333
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.5,
      "trade_id": "PHASE25_00598",
      "entry_time": "2017-08-04 07:57:00-04:00",
      "balance_start_day_pct": 266.9299,
      "equity_low_estimate_pct": 262.0499,
      "closed_pnl_pct": -1.5,
      "open_pnl_proxy_pct": -4.88,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -0.88,
      "mae_r": -3.2533
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 0.85,
      "trade_id": "PHASE25_00715",
      "entry_time": "2018-02-14 07:36:00-05:00",
      "balance_start_day_pct": 181.4075,
      "equity_low_estimate_pct": 176.5634,
      "closed_pnl_pct": -0.85,
      "open_pnl_proxy_pct": -4.8441,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -0.8441,
      "mae_r": -5.6989
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.25,
      "trade_id": "PHASE25_00546",
      "entry_time": "2017-05-12 07:45:00-04:00",
      "balance_start_day_pct": 204.083,
      "equity_low_estimate_pct": 199.583,
      "closed_pnl_pct": -1.25,
      "open_pnl_proxy_pct": -4.5,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -0.5,
      "mae_r": -3.6
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 0.75,
      "trade_id": "PHASE25_02320",
      "entry_time": "2025-01-20 07:27:00-05:00",
      "balance_start_day_pct": 505.0633,
      "equity_low_estimate_pct": 500.5693,
      "closed_pnl_pct": -0.75,
      "open_pnl_proxy_pct": -4.494,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -0.494,
      "mae_r": -5.992
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 0.75,
      "trade_id": "PHASE25_00715",
      "entry_time": "2018-02-14 07:36:00-05:00",
      "balance_start_day_pct": 160.0654,
      "equity_low_estimate_pct": 155.7913,
      "closed_pnl_pct": -0.75,
      "open_pnl_proxy_pct": -4.2742,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -0.2742,
      "mae_r": -5.6989
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.0,
      "trade_id": "PHASE25_02461",
      "entry_time": "2025-08-22 09:36:00-04:00",
      "balance_start_day_pct": 703.9971,
      "equity_low_estimate_pct": 699.7551,
      "closed_pnl_pct": 0.0,
      "open_pnl_proxy_pct": -4.242,
      "daily_loss_limit_pct": 4.0,
      "breach": true,
      "breach_margin_pct": -0.242,
      "mae_r": -4.242
    }
  ]
}

## Max loss 8%
{
  "max_loss_limit_pct": 8.0,
  "equity_balance_floor_pct": 92.0,
  "breach_cases": 0,
  "breaches_by_risk": {},
  "risk_075_maintains_margin": true,
  "risk_100_too_close": true,
  "worst_historical_sequence": [
    {
      "strategy": "PHASE25",
      "profile": "FUNDEDNEXT_STELLAR_LITE_10K_DEFAULT",
      "risk_pct": 1.5,
      "start_month": "2019-12",
      "status": "PASS",
      "phase_reached": "FUNDED_READY",
      "breach_type": "",
      "phase1_pass": true,
      "phase2_pass": true,
      "daily_loss_breach": false,
      "max_loss_breach": false,
      "trades_used": 48,
      "trading_days": 48,
      "days_elapsed": 76,
      "final_return_pct": 4.8,
      "max_dd_pct": -7.8922,
      "worst_daily_loss_pct": -2.0882,
      "max_daily_equity_loss_pct": 2.0882
    },
    {
      "strategy": "PHASE25",
      "profile": "FUNDEDNEXT_STELLAR_LITE_10K_DEFAULT",
      "risk_pct": 1.5,
      "start_month": "2025-02",
      "status": "PASS",
      "phase_reached": "FUNDED_READY",
      "breach_type": "",
      "phase1_pass": true,
      "phase2_pass": true,
      "daily_loss_breach": false,
      "max_loss_breach": false,
      "trades_used": 72,
      "trading_days": 72,
      "days_elapsed": 107,
      "final_return_pct": 5.0211,
      "max_dd_pct": -7.5,
      "worst_daily_loss_pct": -2.4591,
      "max_daily_equity_loss_pct": 2.4591
    },
    {
      "strategy": "PHASE25",
      "profile": "FUNDEDNEXT_STELLAR_LITE_10K_DEFAULT",
      "risk_pct": 1.5,
      "start_month": "2015-10",
      "status": "PASS",
      "phase_reached": "FUNDED_READY",
      "breach_type": "",
      "phase1_pass": true,
      "phase2_pass": true,
      "daily_loss_breach": false,
      "max_loss_breach": false,
      "trades_used": 56,
      "trading_days": 56,
      "days_elapsed": 98,
      "final_return_pct": 6.3,
      "max_dd_pct": -7.4928,
      "worst_daily_loss_pct": -1.8937,
      "max_daily_equity_loss_pct": 1.8937
    },
    {
      "strategy": "PHASE25",
      "profile": "FUNDEDNEXT_STELLAR_LITE_10K_DEFAULT",
      "risk_pct": 1.25,
      "start_month": "2019-08",
      "status": "PASS",
      "phase_reached": "FUNDED_READY",
      "breach_type": "",
      "phase1_pass": true,
      "phase2_pass": true,
      "daily_loss_breach": false,
      "max_loss_breach": false,
      "trades_used": 50,
      "trading_days": 50,
      "days_elapsed": 84,
      "final_return_pct": 4.8963,
      "max_dd_pct": -6.9798,
      "worst_daily_loss_pct": -2.5147,
      "max_daily_equity_loss_pct": 2.5147
    },
    {
      "strategy": "PHASE25",
      "profile": "FUNDEDNEXT_STELLAR_LITE_10K_DEFAULT",
      "risk_pct": 1.25,
      "start_month": "2019-12",
      "status": "PASS",
      "phase_reached": "FUNDED_READY",
      "breach_type": "",
      "phase1_pass": true,
      "phase2_pass": true,
      "daily_loss_breach": false,
      "max_loss_breach": false,
      "trades_used": 49,
      "trading_days": 49,
      "days_elapsed": 79,
      "final_return_pct": 4.0,
      "max_dd_pct": -6.5768,
      "worst_daily_loss_pct": -1.7402,
      "max_daily_equity_loss_pct": 1.7402
    },
    {
      "strategy": "PHASE25",
      "profile": "FUNDEDNEXT_STELLAR_LITE_10K_DEFAULT",
      "risk_pct": 1.5,
      "start_month": "2021-04",
      "status": "PASS",
      "phase_reached": "FUNDED_READY",
      "breach_type": "",
      "phase1_pass": true,
      "phase2_pass": true,
      "daily_loss_breach": false,
      "max_loss_breach": false,
      "trades_used": 40,
      "trading_days": 40,
      "days_elapsed": 70,
      "final_return_pct": 5.3963,
      "max_dd_pct": -6.549,
      "worst_daily_loss_pct": -1.7404,
      "max_daily_equity_loss_pct": 1.7404
    },
    {
      "strategy": "PHASE25",
      "profile": "FUNDEDNEXT_STELLAR_LITE_10K_DEFAULT",
      "risk_pct": 1.25,
      "start_month": "2025-02",
      "status": "PASS",
      "phase_reached": "FUNDED_READY",
      "breach_type": "",
      "phase1_pass": true,
      "phase2_pass": true,
      "daily_loss_breach": false,
      "max_loss_breach": false,
      "trades_used": 74,
      "trading_days": 74,
      "days_elapsed": 114,
      "final_return_pct": 4.9342,
      "max_dd_pct": -6.25,
      "worst_daily_loss_pct": -2.0493,
      "max_daily_equity_loss_pct": 2.0493
    },
    {
      "strategy": "PHASE25",
      "profile": "FUNDEDNEXT_STELLAR_LITE_10K_DEFAULT",
      "risk_pct": 1.25,
      "start_month": "2015-10",
      "status": "PASS",
      "phase_reached": "FUNDED_READY",
      "breach_type": "",
      "phase1_pass": true,
      "phase2_pass": true,
      "daily_loss_breach": false,
      "max_loss_breach": false,
      "trades_used": 59,
      "trading_days": 59,
      "days_elapsed": 103,
      "final_return_pct": 5.25,
      "max_dd_pct": -6.244,
      "worst_daily_loss_pct": -1.5781,
      "max_daily_equity_loss_pct": 1.5781
    },
    {
      "strategy": "PHASE25",
      "profile": "FUNDEDNEXT_STELLAR_LITE_10K_DEFAULT",
      "risk_pct": 1.25,
      "start_month": "2015-09",
      "status": "PASS",
      "phase_reached": "FUNDED_READY",
      "breach_type": "",
      "phase1_pass": true,
      "phase2_pass": true,
      "daily_loss_breach": false,
      "max_loss_breach": false,
      "trades_used": 66,
      "trading_days": 66,
      "days_elapsed": 112,
      "final_return_pct": 4.506,
      "max_dd_pct": -6.244,
      "worst_daily_loss_pct": -1.5781,
      "max_daily_equity_loss_pct": 1.5781
    },
    {
      "strategy": "PHASE25",
      "profile": "FUNDEDNEXT_STELLAR_LITE_10K_DEFAULT",
      "risk_pct": 1.5,
      "start_month": "2018-11",
      "status": "PASS",
      "phase_reached": "FUNDED_READY",
      "breach_type": "",
      "phase1_pass": true,
      "phase2_pass": true,
      "daily_loss_breach": false,
      "max_loss_breach": false,
      "trades_used": 41,
      "trading_days": 41,
      "days_elapsed": 68,
      "final_return_pct": 4.5,
      "max_dd_pct": -6.0,
      "worst_daily_loss_pct": -1.9462,
      "max_daily_equity_loss_pct": 1.9462
    }
  ]
}

## Funded survival
{
  "horizons_months": [
    1,
    3,
    6,
    12
  ],
  "first_payout_cycle_days": 21,
  "subsequent_payout_cycle_days": 14,
  "risk_0.10_1m": {
    "windows": 135,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 0.5443,
    "worst_dd": -0.5,
    "worst_daily_loss": -0.5992,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 94.07,
    "negative_windows": 8
  },
  "risk_0.10_3m": {
    "windows": 133,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 1.642,
    "worst_dd": -0.5584,
    "worst_daily_loss": -0.5992,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.10_6m": {
    "windows": 130,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 3.2829,
    "worst_dd": -0.5584,
    "worst_daily_loss": -0.5992,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.10_12m": {
    "windows": 124,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 6.5385,
    "worst_dd": -0.5584,
    "worst_daily_loss": -0.5992,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.25_1m": {
    "windows": 135,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 1.3608,
    "worst_dd": -1.25,
    "worst_daily_loss": -1.498,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 94.07,
    "negative_windows": 8
  },
  "risk_0.25_3m": {
    "windows": 133,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 4.105,
    "worst_dd": -1.396,
    "worst_daily_loss": -1.498,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.25_6m": {
    "windows": 130,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 8.2073,
    "worst_dd": -1.396,
    "worst_daily_loss": -1.498,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.25_12m": {
    "windows": 124,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 16.3463,
    "worst_dd": -1.396,
    "worst_daily_loss": -1.498,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.35_1m": {
    "windows": 135,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 1.9052,
    "worst_dd": -1.75,
    "worst_daily_loss": -2.0972,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 94.07,
    "negative_windows": 8
  },
  "risk_0.35_3m": {
    "windows": 133,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 5.747,
    "worst_dd": -1.9544,
    "worst_daily_loss": -2.0972,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.35_6m": {
    "windows": 130,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 11.4903,
    "worst_dd": -1.9544,
    "worst_daily_loss": -2.0972,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.35_12m": {
    "windows": 124,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 22.8849,
    "worst_dd": -1.9544,
    "worst_daily_loss": -2.0972,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.50_1m": {
    "windows": 135,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 2.7217,
    "worst_dd": -2.5,
    "worst_daily_loss": -2.996,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 94.07,
    "negative_windows": 8
  },
  "risk_0.50_3m": {
    "windows": 133,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 8.21,
    "worst_dd": -2.7919,
    "worst_daily_loss": -2.996,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.50_6m": {
    "windows": 130,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 16.4147,
    "worst_dd": -2.7919,
    "worst_daily_loss": -2.996,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.50_12m": {
    "windows": 124,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 32.6926,
    "worst_dd": -2.7919,
    "worst_daily_loss": -2.996,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.60_1m": {
    "windows": 135,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 3.266,
    "worst_dd": -3.0,
    "worst_daily_loss": -3.5952,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 94.07,
    "negative_windows": 8
  },
  "risk_0.60_3m": {
    "windows": 133,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 9.8519,
    "worst_dd": -3.3503,
    "worst_daily_loss": -3.5952,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.60_6m": {
    "windows": 130,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 19.6976,
    "worst_dd": -3.3503,
    "worst_daily_loss": -3.5952,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.60_12m": {
    "windows": 124,
    "survival_probability": 100.0,
    "breach_probability": 0.0,
    "expected_return": 39.2312,
    "worst_dd": -3.3503,
    "worst_daily_loss": -3.5952,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 100.0,
    "negative_windows": 0
  },
  "risk_0.75_1m": {
    "windows": 135,
    "survival_probability": 98.52,
    "breach_probability": 1.48,
    "expected_return": 4.0803,
    "worst_dd": -3.75,
    "worst_daily_loss": -4.494,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 92.59,
    "negative_windows": 8
  },
  "risk_0.75_3m": {
    "windows": 133,
    "survival_probability": 95.49,
    "breach_probability": 4.51,
    "expected_return": 12.2121,
    "worst_dd": -4.1879,
    "worst_daily_loss": -4.494,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 95.49,
    "negative_windows": 0
  },
  "risk_0.75_6m": {
    "windows": 130,
    "survival_probability": 90.77,
    "breach_probability": 9.23,
    "expected_return": 23.9478,
    "worst_dd": -4.1879,
    "worst_daily_loss": -4.494,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 90.77,
    "negative_windows": 0
  },
  "risk_0.75_12m": {
    "windows": 124,
    "survival_probability": 80.65,
    "breach_probability": 19.35,
    "expected_return": 45.2549,
    "worst_dd": -4.1879,
    "worst_daily_loss": -4.494,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 80.65,
    "negative_windows": 0
  },
  "risk_0.85_1m": {
    "windows": 135,
    "survival_probability": 98.52,
    "breach_probability": 1.48,
    "expected_return": 4.6243,
    "worst_dd": -4.25,
    "worst_daily_loss": -5.0932,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 92.59,
    "negative_windows": 8
  },
  "risk_0.85_3m": {
    "windows": 133,
    "survival_probability": 95.49,
    "breach_probability": 4.51,
    "expected_return": 13.8404,
    "worst_dd": -4.7463,
    "worst_daily_loss": -5.0932,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 95.49,
    "negative_windows": 0
  },
  "risk_0.85_6m": {
    "windows": 130,
    "survival_probability": 90.77,
    "breach_probability": 9.23,
    "expected_return": 27.1408,
    "worst_dd": -4.7463,
    "worst_daily_loss": -5.0932,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 90.77,
    "negative_windows": 0
  },
  "risk_0.85_12m": {
    "windows": 124,
    "survival_probability": 80.65,
    "breach_probability": 19.35,
    "expected_return": 51.2889,
    "worst_dd": -4.7463,
    "worst_daily_loss": -5.0932,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 80.65,
    "negative_windows": 0
  },
  "risk_1.00_1m": {
    "windows": 135,
    "survival_probability": 97.04,
    "breach_probability": 2.96,
    "expected_return": 5.4093,
    "worst_dd": -5.0,
    "worst_daily_loss": -5.992,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 91.11,
    "negative_windows": 8
  },
  "risk_1.00_3m": {
    "windows": 133,
    "survival_probability": 90.98,
    "breach_probability": 9.02,
    "expected_return": 16.0628,
    "worst_dd": -5.5839,
    "worst_daily_loss": -5.992,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 90.98,
    "negative_windows": 0
  },
  "risk_1.00_6m": {
    "windows": 130,
    "survival_probability": 81.54,
    "breach_probability": 18.46,
    "expected_return": 30.9549,
    "worst_dd": -5.5839,
    "worst_daily_loss": -5.992,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 81.54,
    "negative_windows": 0
  },
  "risk_1.00_12m": {
    "windows": 124,
    "survival_probability": 73.39,
    "breach_probability": 26.61,
    "expected_return": 57.422,
    "worst_dd": -5.5839,
    "worst_daily_loss": -5.992,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 73.39,
    "negative_windows": 0
  },
  "risk_1.25_1m": {
    "windows": 135,
    "survival_probability": 94.81,
    "breach_probability": 5.19,
    "expected_return": 6.6936,
    "worst_dd": -6.25,
    "worst_daily_loss": -7.49,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 88.89,
    "negative_windows": 9
  },
  "risk_1.25_3m": {
    "windows": 133,
    "survival_probability": 84.21,
    "breach_probability": 15.79,
    "expected_return": 19.3718,
    "worst_dd": -6.9798,
    "worst_daily_loss": -7.49,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 84.21,
    "negative_windows": 1
  },
  "risk_1.25_6m": {
    "windows": 130,
    "survival_probability": 71.54,
    "breach_probability": 28.46,
    "expected_return": 36.2012,
    "worst_dd": -6.9798,
    "worst_daily_loss": -7.49,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 71.54,
    "negative_windows": 1
  },
  "risk_1.25_12m": {
    "windows": 124,
    "survival_probability": 58.06,
    "breach_probability": 41.94,
    "expected_return": 64.1931,
    "worst_dd": -6.9798,
    "worst_daily_loss": -7.49,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 58.06,
    "negative_windows": 1
  },
  "risk_1.50_1m": {
    "windows": 135,
    "survival_probability": 94.81,
    "breach_probability": 5.19,
    "expected_return": 8.0323,
    "worst_dd": -7.5,
    "worst_daily_loss": -8.988,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 88.89,
    "negative_windows": 9
  },
  "risk_1.50_3m": {
    "windows": 133,
    "survival_probability": 84.21,
    "breach_probability": 15.79,
    "expected_return": 23.2461,
    "worst_dd": -8.3758,
    "worst_daily_loss": -8.988,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 84.21,
    "negative_windows": 1
  },
  "risk_1.50_6m": {
    "windows": 130,
    "survival_probability": 71.54,
    "breach_probability": 28.46,
    "expected_return": 43.4415,
    "worst_dd": -8.3758,
    "worst_daily_loss": -8.988,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 71.54,
    "negative_windows": 1
  },
  "risk_1.50_12m": {
    "windows": 124,
    "survival_probability": 58.06,
    "breach_probability": 41.94,
    "expected_return": 77.0317,
    "worst_dd": -8.3758,
    "worst_daily_loss": -8.988,
    "payout_cycle_compatible": true,
    "first_payout_21day_positive_survival": 58.06,
    "negative_windows": 1
  },
  "recommended_funded_risk": 0.5,
  "risk_075_funded_defensible": false
}

## News/weekend
{
  "challenge_news_allowed": true,
  "funded_news_allowed_with_profit_split_rule": true,
  "challenge_weekend_holding_allowed": true,
  "funded_weekend_holding_not_allowed": true,
  "phase25_intraday_policy_claim": "Phase25 is intended as an intraday/daytime line, but the physical historical ledger shows weekend-crossing exits.",
  "weekend_cross_cases": 31,
  "news_profit_split_warning_cases": 0,
  "news_fortress_avoids_problem": true,
  "funded_weekend_rule_status": "WARNING_BLOCKER_UNTIL_OPERATIONAL_CLOSE_POLICY_AUDITED",
  "status": "WARNING_FUNDED_WEEKEND_HOLDING_CASES"
}

## Comparacion contra FTMO
{
  "stellar_lite_better_than_ftmo_1step_for_phase25": true,
  "stellar_lite_better_than_ftmo_2step": false,
  "good_first_small_evaluation": "PAPER_OR_FREE_TRIAL_FIRST; real purchase not recommended from price alone.",
  "risk_075_defensible": false,
  "risk_050_healthier": true,
  "funded_survival": {
    "0.50_12m": {
      "windows": 124,
      "survival_probability": 100.0,
      "breach_probability": 0.0,
      "expected_return": 32.6926,
      "worst_dd": -2.7919,
      "worst_daily_loss": -2.996,
      "payout_cycle_compatible": true,
      "first_payout_21day_positive_survival": 100.0,
      "negative_windows": 0
    },
    "0.75_12m": {
      "windows": 124,
      "survival_probability": 80.65,
      "breach_probability": 19.35,
      "expected_return": 45.2549,
      "worst_dd": -4.1879,
      "worst_daily_loss": -4.494,
      "payout_cycle_compatible": true,
      "first_payout_21day_positive_survival": 80.65,
      "negative_windows": 0
    }
  },
  "scorecard_rows": [
    {
      "risk_pct": 0.5,
      "fundednext_combined_pass": 97.79,
      "fundednext_daily_breach": 0.0,
      "fundednext_max_breach": 0.0,
      "ftmo1_pass": 97.79,
      "ftmo1_daily_breach": 0.0,
      "ftmo2_challenge_pass": 97.79,
      "ftmo2_daily_breach": 0.0,
      "daily_loss_margin_winner": "FTMO_2_STEP"
    },
    {
      "risk_pct": 0.75,
      "fundednext_combined_pass": 93.38,
      "fundednext_daily_breach": 5.15,
      "fundednext_max_breach": 0.0,
      "ftmo1_pass": 91.91,
      "ftmo1_daily_breach": 6.62,
      "ftmo2_challenge_pass": 98.53,
      "ftmo2_daily_breach": 0.0,
      "daily_loss_margin_winner": "FTMO_2_STEP"
    },
    {
      "risk_pct": 1.0,
      "fundednext_combined_pass": 91.91,
      "fundednext_daily_breach": 6.62,
      "fundednext_max_breach": 0.0,
      "ftmo1_pass": 88.24,
      "ftmo1_daily_breach": 10.29,
      "ftmo2_challenge_pass": 94.85,
      "ftmo2_daily_breach": 3.68,
      "daily_loss_margin_winner": "FTMO_2_STEP"
    }
  ]
}

## Riesgo recomendado
- Risk recommended: 0.50%.
- 0.75% is not a base recommendation unless future forward evidence validates it.
- 1.00% is not recommended.

## Veredicto final
PHASE32B_FUNDEDNEXT_STELLAR_LITE_SUPPORTED_WITH_WARNINGS

## Siguiente paso unico
Run a Phase32 paper/free-trial decision checkpoint for Stellar Lite at 0.50%, not real purchase.
