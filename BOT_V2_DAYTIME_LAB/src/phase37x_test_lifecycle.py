from __future__ import annotations
import json
from datetime import datetime, time
from phase37x_session_lifecycle import get_session_state

def test_lifecycle():
    # We can't easily mock datetime.now() without external libs like freezegun
    # But we can verify the current state logic
    now = datetime.now()
    state = get_session_state(position_open=False)
    print(f"Current Local Time: {now}")
    print(f"Detected State (Flat): {state}")
    
    state_pos = get_session_state(position_open=True)
    print(f"Detected State (Open Position): {state_pos}")

if __name__ == "__main__":
    test_lifecycle()
