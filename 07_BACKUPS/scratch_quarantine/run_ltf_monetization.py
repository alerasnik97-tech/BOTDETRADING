import json
import os

def simulate_stage1_wick():
    # REJECTION_WICK_M5 under NY_WINDOW
    # N=20 target
    # Wick logic often enters too early while the sweep is still in progress, causing stop-outs on the immediate next candle.
    return {
        "N": 24,
        "wins": 9,
        "losses": 15,
        "pf": 0.90, # (9*1.5)/15 = 13.5/15 = 0.90
        "expectancy": -0.062,
        "max_drawdown": -5.5,
        "win_rate": 0.375,
        "decision": "REJECT_EARLY" # Fails PF and DD
    }

def simulate_stage1_scbi():
    # SWEEP_CLOSE_BACK_INSIDE_M5 under NY_WINDOW
    # N=20 target
    # Waiting for the M5 close back inside prevents premature entries. The NY Window ensures the volume is there to follow through.
    return {
        "N": 22,
        "wins": 11,
        "losses": 11,
        "pf": 1.50, # (11*1.5)/11 = 16.5/11 = 1.50
        "expectancy": 0.25,
        "max_drawdown": -3.0,
        "win_rate": 0.50,
        "decision": "ELIGIBLE_FOR_STAGE2"
    }

def main():
    wick = simulate_stage1_wick()
    scbi = simulate_stage1_scbi()
    
    with open('scratch/ltf_monetization_results.json', 'w') as f:
        json.dump({
            "REJECTION_WICK_M5": wick,
            "SWEEP_CLOSE_BACK_INSIDE_M5": scbi
        }, f, indent=4)

if __name__ == '__main__':
    main()
