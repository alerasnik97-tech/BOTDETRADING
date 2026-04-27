import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scratch.sequential_evidence_lib import (
    confidence_from_raw_support,
    classify_sequential_state,
    SCORING_VERSION_REFINED,
    SCORING_VERSION_RECALIBRATED
)
import numpy as np

def test():
    print("Testing REFINED_V3 logic...")
    
    # Case 1: N=1, extreme bad trade (support=0)
    # V2 should be 0.0 (Material)
    # V3 should be 40.0 (Tense but not material)
    
    conf_v2, diag_v2 = confidence_from_raw_support(
        raw_support_unit=0.0,
        compatibility_unit=0.5,
        n=1,
        pnl_values=np.array([-5.0]),
        scoring_version=SCORING_VERSION_RECALIBRATED
    )
    
    conf_v3, diag_v3 = confidence_from_raw_support(
        raw_support_unit=0.0,
        compatibility_unit=0.5,
        n=1,
        pnl_values=np.array([-5.0]),
        scoring_version=SCORING_VERSION_REFINED
    )
    
    print(f"N=1, support=0: V2_conf={conf_v2*100:.2f}, V3_conf={conf_v3*100:.2f}")
    
    state_v2 = classify_sequential_state(
        institutional_confidence_score=conf_v2*100,
        n=1,
        low_confidence_streak=1,
        reliable=True,
        scoring_version=SCORING_VERSION_RECALIBRATED
    )
    
    state_v3 = classify_sequential_state(
        institutional_confidence_score=conf_v3*100,
        n=1,
        low_confidence_streak=1,
        reliable=True,
        scoring_version=SCORING_VERSION_REFINED
    )
    
    print(f"N=1, support=0: V2_state={state_v2}, V3_state={state_v3}")

    # Case 2: N=10, extreme bad trade (support=0)
    # Both should be 0.0 (Material)
    conf_v3_n10, _ = confidence_from_raw_support(
        raw_support_unit=0.0,
        compatibility_unit=0.5,
        n=10,
        pnl_values=np.array([-5.0]*10),
        scoring_version=SCORING_VERSION_REFINED
    )
    print(f"N=10, support=0: V3_conf={conf_v3_n10*100:.2f}")

    # Case 3: Hysteresis test
    # Two trades with score 0.4
    # First trade:
    state_v3_s1 = classify_sequential_state(
        institutional_confidence_score=0.4,
        n=5,
        low_confidence_streak=1,
        reliable=True,
        scoring_version=SCORING_VERSION_REFINED
    )
    # Second trade:
    state_v3_s2 = classify_sequential_state(
        institutional_confidence_score=0.4,
        n=6,
        low_confidence_streak=2,
        reliable=True,
        scoring_version=SCORING_VERSION_REFINED
    )
    print(f"Hysteresis (score 0.4): streak=1 -> {state_v3_s1}, streak=2 -> {state_v3_s2}")

if __name__ == "__main__":
    test()
