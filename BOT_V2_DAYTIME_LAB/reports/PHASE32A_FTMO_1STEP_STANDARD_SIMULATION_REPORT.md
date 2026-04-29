# PHASE32A FTMO 1-STEP STANDARD SIMULATION REPORT

## Objetivo
Simulate whether Phase25 authority supports FTMO 1-Step Standard.

## Estrategia simulada
- PHASE25_AUTHORITY only.
- TP1.4 / BE0.4 / BF70.
- No shadow candidate.

## Reglas FTMO 1-Step
- Profit Target: 10%.
- Max Daily Loss: 3%.
- Max Loss: 10% trailing EOD.
- Best Day Rule: 50%.
- Standard only; Swing unavailable for 1-Step.

## Resultados clave
- 0.50%: pass 97.79%, daily breach 0.0%, BDR block 0.0%.
- 0.75%: pass 91.91%, daily breach 6.62%, BDR block 0.0%.
- 1.00%: pass 88.24%, daily breach 10.29%, BDR block 0.0%.

## Daily loss 3%
{
  "daily_loss_limit_pct": 3.0,
  "intraday_equity_mode": "mae_proxy",
  "breach_cases": 48,
  "breaches_by_risk": {
    "0.6": 2,
    "0.75": 4,
    "1.0": 8,
    "1.25": 14,
    "1.5": 20
  },
  "risk_075_supports_daily_loss_3pct": false,
  "risk_050_more_prudent": true,
  "risk_100_discarded_as_base": true,
  "pure_sl_streak_4_threatens_3pct": "At 0.75% four consecutive pure SL would equal 3.0% before MAE/cost buffer, so 0.75% is a ceiling; at 0.50% four SL equals 2.0%.",
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -5.988,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -5.5484,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -4.49,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -4.1237,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -3.3631,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -3.1429,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -2.992,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -2.6989,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -2.4,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -2.3025,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -2.119,
      "mae_r": -4.0952
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -2.027,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -2.0,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -1.88,
      "mae_r": -3.2533
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -1.5,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -1.494,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -1.2742,
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
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -1.242,
      "mae_r": -4.242
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.25,
      "trade_id": "PHASE25_01305",
      "entry_time": "2020-09-16 13:24:00-04:00",
      "balance_start_day_pct": 466.8128,
      "equity_low_estimate_pct": 462.6236,
      "closed_pnl_pct": 0.0,
      "open_pnl_proxy_pct": -4.1892,
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -1.1892,
      "mae_r": -3.3514
    },
    {
      "strategy": "PHASE25",
      "risk_pct": 1.25,
      "trade_id": "PHASE25_00469",
      "entry_time": "2017-01-06 07:57:00-05:00",
      "balance_start_day_pct": 181.6473,
      "equity_low_estimate_pct": 177.4807,
      "closed_pnl_pct": 0.0,
      "open_pnl_proxy_pct": -4.1667,
      "daily_loss_limit_pct": 3.0,
      "breach": true,
      "breach_margin_pct": -1.1667,
      "mae_r": -3.3333
    }
  ]
}

## Best Day Rule
{
  "modeled_rule": "Best day must be <= 50% of positive days profit; modeled as approval blocker, not immediate breach.",
  "is_problem_for_phase25": false,
  "cases": 0,
  "cases_by_risk": {},
  "appears_at_risk": null,
  "affects_075": false,
  "makes_1step_worse_than_2step": false,
  "note": "The rule usually delays pass approval after a large winning day; it does not create a monetary breach by itself."
}

## Standard vs Swing
{
  "ftmo_1step_allows_swing": false,
  "standard_enough_for_phase25_evaluation": true,
  "strategy_holds_overnight_weekend": false,
  "strategy_trades_near_news": "News Fortress is fail-closed; evaluation news restriction not applied per official FAQ, but funded Standard still needs manual operating review.",
  "news_fortress_covers_restrictions": "Likely yes for Phase25 intent, but funded account rule review remains mandatory.",
  "standard_evaluation_restrictions_relevant": false,
  "standard_funded_news_risk": "WARNING_ONLY; Standard funded restrictions apply after FTMO Account.",
  "swing_2step_future_cleaner": true,
  "recommended_decision": "FTMO_2_STEP_SWING_PREFERRED_FOR_FUTURE_FUNDED_CLEANLINESS; FTMO_1_STEP_STANDARD_ONLY_WITH_CONDITIONS"
}

## Comparacion 1-Step vs 2-Step
{
  "buy_1step": "NO_AS_BASE; only paper/free-trial planning with strict conditions.",
  "prefer_2step": true,
  "capital_preservation_winner": "FTMO_2_STEP_SWING_OR_2_STEP_STANDARD",
  "phase25_compatibility": "1-Step can work at conservative risk but has tighter daily loss and Best Day Rule complexity.",
  "one_step_050": {
    "strategy": "PHASE25",
    "profile": "FTMO_1_STEP_STANDARD_DEFAULT",
    "risk_pct": 0.5,
    "windows": 136,
    "pass_probability": 97.79,
    "fail_probability": 0.0,
    "daily_loss_breach_probability": 0.0,
    "max_loss_breach_probability": 0.0,
    "best_day_rule_violation_probability": 0.0,
    "average_days_to_pass": 112.2,
    "median_days_to_pass": 107.0,
    "average_trades_to_pass": 71.62,
    "median_trades_to_pass": 69.0,
    "worst_historical_window": "2026-04",
    "best_historical_window": "2026-04",
    "max_dd": -2.7919,
    "max_daily_equity_loss": 2.996,
    "recommended_risk": "ACCEPTABLE"
  },
  "one_step_075": {
    "strategy": "PHASE25",
    "profile": "FTMO_1_STEP_STANDARD_DEFAULT",
    "risk_pct": 0.75,
    "windows": 136,
    "pass_probability": 91.91,
    "fail_probability": 6.62,
    "daily_loss_breach_probability": 6.62,
    "max_loss_breach_probability": 0.0,
    "best_day_rule_violation_probability": 0.0,
    "average_days_to_pass": 72.19,
    "median_days_to_pass": 68.0,
    "average_trades_to_pass": 46.1,
    "median_trades_to_pass": 45.0,
    "worst_historical_window": "2026-04",
    "best_historical_window": "2017-08",
    "max_dd": -4.1879,
    "max_daily_equity_loss": 4.494,
    "recommended_risk": "NOT_BASE"
  },
  "standard_vs_swing": {
    "ftmo_1step_allows_swing": false,
    "standard_enough_for_phase25_evaluation": true,
    "strategy_holds_overnight_weekend": false,
    "strategy_trades_near_news": "News Fortress is fail-closed; evaluation news restriction not applied per official FAQ, but funded Standard still needs manual operating review.",
    "news_fortress_covers_restrictions": "Likely yes for Phase25 intent, but funded account rule review remains mandatory.",
    "standard_evaluation_restrictions_relevant": false,
    "standard_funded_news_risk": "WARNING_ONLY; Standard funded restrictions apply after FTMO Account.",
    "swing_2step_future_cleaner": true,
    "recommended_decision": "FTMO_2_STEP_SWING_PREFERRED_FOR_FUTURE_FUNDED_CLEANLINESS; FTMO_1_STEP_STANDARD_ONLY_WITH_CONDITIONS"
  },
  "scorecard_rows": [
    {
      "risk_pct": 0.5,
      "one_step_pass_probability": 97.79,
      "one_step_daily_loss_breach": 0.0,
      "one_step_max_loss_breach": 0.0,
      "one_step_bdr_block": 0.0,
      "two_step_challenge_pass_probability": 97.79,
      "two_step_daily_loss_breach": 0.0,
      "two_step_max_loss_breach": 0.0,
      "safety_margin_winner": "2-Step"
    },
    {
      "risk_pct": 0.75,
      "one_step_pass_probability": 91.91,
      "one_step_daily_loss_breach": 6.62,
      "one_step_max_loss_breach": 0.0,
      "one_step_bdr_block": 0.0,
      "two_step_challenge_pass_probability": 98.53,
      "two_step_daily_loss_breach": 0.0,
      "two_step_max_loss_breach": 0.0,
      "safety_margin_winner": "2-Step"
    },
    {
      "risk_pct": 1.0,
      "one_step_pass_probability": 88.24,
      "one_step_daily_loss_breach": 10.29,
      "one_step_max_loss_breach": 0.0,
      "one_step_bdr_block": 0.0,
      "two_step_challenge_pass_probability": 94.85,
      "two_step_daily_loss_breach": 3.68,
      "two_step_max_loss_breach": 0.0,
      "safety_margin_winner": "2-Step"
    }
  ]
}

## Riesgo recomendado
- Risk recommended: 0.50%.
- Max not exceed: 0.75%.
- 1.00% not recommended.

## Veredicto final
PHASE32A_FTMO_1STEP_SUPPORTED_WITH_WARNINGS

## Siguiente paso unico
Run Phase32 paper/demo discipline on 2-Step preferred path, while using 1-Step only as paper/free-trial scenario if desired.
