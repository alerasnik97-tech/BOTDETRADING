# PHASE29 WR + LOSS STREAK COMPRESSION REPORT

## Objective
Research shadow only. Improve WR and compress canonical non-win streak without weakening Phase25 governance.

## Baseline Phase25
- Sample: 2625
- PF: 2.793
- Expectancy: 0.2809
- WR: 32.53
- Max DD: -5.584
- Max loss streak: 14
- Trades/month: 19.36
- Months <15 trades: 0

## Diagnostics
- Streaks >=5: 120
- Streaks >=8: 30
- Streaks >=10: 15
- Streaks >=12: 3
- BE prevented original SL proxy: 556
- BE prevented TP proxy: 557

## Hypotheses
- Single hypotheses tested: 23
- Limited combinations tested: 5

## Candidate Selection
- Verdict: PHASE29_BALANCED_WR_STREAK_IMPROVEMENT_FOUND
- Best WR: TP1.2_BE0.5_BF65
- Best streak compression: TP1.4_BE0.5_BF70
- Best balanced: TP1.4_BE0.5_BF70
- Phase25 remains authority. No automatic replacement.

## WR 50+
- Healthy WR 50+ viable: False
- Diagnostic: WR 50+ rows are rejected unless PF/DD/frequency survive.

## Max Loss Streak <=12
- Healthy <=12 viable: True

## Walk-forward / Holdout
{
  "BEST_BALANCED_CANDIDATE": {
    "test_passes": 5,
    "test_total": 6,
    "failed_splits": [
      "2019_2022__2023_2024__2025"
    ],
    "holdout_2026_pass": true
  },
  "BEST_STREAK_COMPRESSION_CANDIDATE": {
    "test_passes": 5,
    "test_total": 6,
    "failed_splits": [
      "2019_2022__2023_2024__2025"
    ],
    "holdout_2026_pass": true
  },
  "BEST_WR_CANDIDATE": {
    "test_passes": 5,
    "test_total": 6,
    "failed_splits": [
      "2019_2022__2023_2024__2025"
    ],
    "holdout_2026_pass": true
  },
  "PHASE25_BASELINE": {
    "test_passes": 5,
    "test_total": 6,
    "failed_splits": [
      "2019_2022__2023_2024__2025"
    ],
    "holdout_2026_pass": true
  }
}

## Cost Stress
{
  "BEST_BALANCED_CANDIDATE": {
    "slippage": {
      "pf_lt_2_at": 1.5,
      "pf_lt_1_5_at": null,
      "expectancy_le_0_at": null
    },
    "spread_add": {
      "pf_lt_2_at": 1.5,
      "pf_lt_1_5_at": null,
      "expectancy_le_0_at": null
    }
  },
  "BEST_STREAK_COMPRESSION_CANDIDATE": {
    "slippage": {
      "pf_lt_2_at": 1.5,
      "pf_lt_1_5_at": null,
      "expectancy_le_0_at": null
    },
    "spread_add": {
      "pf_lt_2_at": 1.5,
      "pf_lt_1_5_at": null,
      "expectancy_le_0_at": null
    }
  },
  "BEST_WR_CANDIDATE": {
    "slippage": {
      "pf_lt_2_at": 1.5,
      "pf_lt_1_5_at": null,
      "expectancy_le_0_at": null
    },
    "spread_add": {
      "pf_lt_2_at": 1.5,
      "pf_lt_1_5_at": null,
      "expectancy_le_0_at": null
    }
  },
  "PHASE25_BASELINE": {
    "slippage": {
      "pf_lt_2_at": 1.5,
      "pf_lt_1_5_at": null,
      "expectancy_le_0_at": null
    },
    "spread_add": {
      "pf_lt_2_at": 1.5,
      "pf_lt_1_5_at": null,
      "expectancy_le_0_at": null
    }
  }
}

## Forensic Safety
{
  "all_clear": true,
  "checks": [
    {
      "candidate": "PHASE25_BASELINE",
      "news_violations": 0,
      "data_mask_violations": 0,
      "trades_without_sl": 0,
      "trades_without_tp": 0,
      "out_of_hours": 0,
      "duplicate_trades": 0,
      "overlapping_illegal_trades": 0,
      "lookahead_detected": 0,
      "impossible_fills": 0,
      "wrong_bid_ask_side": 0,
      "same_bar_logic_conservative": true,
      "forced_close_correct": true,
      "uses_m5_for_m3": false,
      "uses_uncertified_data": false
    },
    {
      "candidate": "BEST_WR_CANDIDATE",
      "news_violations": 0,
      "data_mask_violations": 0,
      "trades_without_sl": 0,
      "trades_without_tp": 0,
      "out_of_hours": 0,
      "duplicate_trades": 0,
      "overlapping_illegal_trades": 0,
      "lookahead_detected": 0,
      "impossible_fills": 0,
      "wrong_bid_ask_side": 0,
      "same_bar_logic_conservative": true,
      "forced_close_correct": true,
      "uses_m5_for_m3": false,
      "uses_uncertified_data": false
    },
    {
      "candidate": "BEST_STREAK_COMPRESSION_CANDIDATE",
      "news_violations": 0,
      "data_mask_violations": 0,
      "trades_without_sl": 0,
      "trades_without_tp": 0,
      "out_of_hours": 0,
      "duplicate_trades": 0,
      "overlapping_illegal_trades": 0,
      "lookahead_detected": 0,
      "impossible_fills": 0,
      "wrong_bid_ask_side": 0,
      "same_bar_logic_conservative": true,
      "forced_close_correct": true,
      "uses_m5_for_m3": false,
      "uses_uncertified_data": false
    },
    {
      "candidate": "BEST_BALANCED_CANDIDATE",
      "news_violations": 0,
      "data_mask_violations": 0,
      "trades_without_sl": 0,
      "trades_without_tp": 0,
      "out_of_hours": 0,
      "duplicate_trades": 0,
      "overlapping_illegal_trades": 0,
      "lookahead_detected": 0,
      "impossible_fills": 0,
      "wrong_bid_ask_side": 0,
      "same_bar_logic_conservative": true,
      "forced_close_correct": true,
      "uses_m5_for_m3": false,
      "uses_uncertified_data": false
    }
  ],
  "violation_count": 0
}

## Final Verdict
PHASE29_BALANCED_WR_STREAK_IMPROVEMENT_FOUND
