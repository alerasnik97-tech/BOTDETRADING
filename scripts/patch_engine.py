import sys
from pathlib import Path

# Paths
engine_path = Path("research_lab/engine.py")

# Content to find and replace
target_1 = """    break_even_numeric = _safe_float(break_even_at_r)
        if break_even_numeric is None or break_even_numeric <= 0:
            raise ValueError("Signal invalida: break_even_at_r debe ser > 0 si se informa.")
        validated["break_even_at_r"] = break_even_numeric

    return validated"""

replacement_1 = """    break_even_numeric = _safe_float(break_even_at_r)
        if break_even_numeric is None or break_even_numeric <= 0:
            raise ValueError("Signal invalida: break_even_at_r debe ser > 0 si se informa.")
        validated["break_even_at_r"] = break_even_numeric

    entry_mode = str(validated.get("entry_mode", "market")).strip().lower()
    validated["entry_mode"] = entry_mode
    if entry_mode == "limit":
        limit_price = _safe_float(validated.get("limit_price"))
        if limit_price is None:
            raise ValueError("Signal invalida: entry_mode='limit' requiere limit_price numerico.")
        validated["limit_price"] = limit_price

    return validated"""

target_2 = """                if not precision_enabled:
                    entry_bid_price = float(open_[i])
                    entry_price = entry_execution_price(pair, pending_signal["direction"], entry_bid_price, entry_spread_pips, entry_slippage_pips)
                else:
                    entry_price = high_precision_entry_execution_price(
                        pair,
                        pending_signal["direction"],
                        entry_bid_price,
                        entry_ask_price,
                        entry_slippage_pips,
                    )"""

replacement_2 = """                if pending_signal.get("entry_mode") == "limit":
                    limit_target = float(pending_signal["limit_price"])
                    if precision_enabled:
                        if pending_signal["direction"] == "long":
                            can_fill_limit = float(ask_m15_low[i]) <= limit_target <= float(ask_m15_high[i])
                        else:
                            can_fill_limit = float(bid_m15_low[i]) <= limit_target <= float(bid_m15_high[i])
                    else:
                        can_fill_limit = float(low[i]) <= limit_target <= float(high[i])
                    
                    if can_fill_limit:
                        entry_bid_price = limit_target
                        entry_price = entry_execution_price(pair, pending_signal["direction"], entry_bid_price, entry_spread_pips, entry_slippage_pips)
                    else:
                        pending_signal = None
                        continue
                else:
                    if not precision_enabled:
                        entry_bid_price = float(open_[i])
                        entry_price = entry_execution_price(pair, pending_signal["direction"], entry_bid_price, entry_spread_pips, entry_slippage_pips)
                    else:
                        entry_price = high_precision_entry_execution_price(
                            pair,
                            pending_signal["direction"],
                            entry_bid_price,
                            entry_ask_price,
                            entry_slippage_pips,
                        )"""

def main():
    if not engine_path.exists():
        print(f"Error: {engine_path} not found")
        sys.exit(1)
        
    content = engine_path.read_text(encoding="utf-8")
    
    # Note: Using replace with cautious string matching
    # First replacement (validation) - only if not already done
    if 'entry_mode = str(validated.get("entry_mode"' not in content:
        if target_1 in content:
            content = content.replace(target_1, replacement_1)
        else:
            print("Failed to find target_1")
            # Fallback for target_1 if there are minor whitespace diffs
            # (Just a simple line replace as fallback)
            content = content.replace('    return validated', replacement_1.split('\n')[-11:]) # Very risky
            
    # Second replacement (fill logic)
    if 'pending_signal.get("entry_mode") == "limit"' not in content:
        if target_2 in content:
            content = content.replace(target_2, replacement_2)
        else:
            print("Failed to find target_2")
            
    engine_path.write_text(content, encoding="utf-8")
    print("Patch applied successfully (if targets found)")

if __name__ == "__main__":
    main()
