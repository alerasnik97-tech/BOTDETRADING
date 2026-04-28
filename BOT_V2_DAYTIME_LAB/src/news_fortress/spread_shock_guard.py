
class SpreadShockGuard:
    def __init__(self, config=None):
        self.config = config or {
            "max_spread_pips": 1.5,
            "shock_multiplier": 2.0
        }

    def evaluate_spread(self, current_spread_pips, median_spread_pips=None):
        if current_spread_pips > self.config['max_spread_pips']:
            return False, f"SPREAD_TOO_HIGH: {current_spread_pips} > {self.config['max_spread_pips']}"
            
        if median_spread_pips:
            threshold = median_spread_pips * self.config['shock_multiplier']
            if current_spread_pips > threshold:
                return False, f"SPREAD_SHOCK_DETECTED: {current_spread_pips} > {threshold} (Median: {median_spread_pips})"
                
        return True, "SPREAD_OK"
