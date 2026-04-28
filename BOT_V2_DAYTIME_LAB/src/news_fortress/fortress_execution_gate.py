
from .news_fortress_gate import NewsFortressGate

class FortressExecutionGate:
    def __init__(self, news_gate, other_gates=None):
        self.news_gate = news_gate
        self.other_gates = other_gates or []

    def evaluate_full_permission(self, timestamp_utc, symbol="EURUSD", context=None):
        # 1. News Gate (The most critical one)
        news_allow, news_reason = self.news_gate.evaluate_trading_permission(timestamp_utc, symbol)
        if not news_allow:
            return False, news_reason, "NEWS_GATE"

        # 2. Risk Check (SL/TP required)
        if context and not context.get('has_sl'):
            return False, "BLOCK: Trading without SL is strictly prohibited.", "RISK_GATE"
        if context and not context.get('has_tp'):
            return False, "BLOCK: Trading without TP is strictly prohibited.", "RISK_GATE"

        # 3. Other gates (Time, Spread, etc. can be added here)
        for gate in self.other_gates:
            allow, reason = gate.evaluate(timestamp_utc, symbol, context)
            if not allow:
                return False, f"BLOCK: {reason}", "OTHER_GATE"

        return True, "ALLOW: All gates passed.", "EXECUTION_GATE"
