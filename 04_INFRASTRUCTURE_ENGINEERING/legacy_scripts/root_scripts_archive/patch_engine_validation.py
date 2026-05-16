from pathlib import Path

engine_path = Path("research_lab/engine.py")

target = """                signal = validate_signal_risk_contract(signal, signal_price=float(close[i]), engine_config=engine_config)"""

replacement = """                effective_signal_price = float(signal.get("limit_price", close[i])) if signal.get("entry_mode") == "limit" else float(close[i])
                signal = validate_signal_risk_contract(signal, signal_price=effective_signal_price, engine_config=engine_config)"""

def main():
    content = engine_path.read_text(encoding="utf-8")
    if target in content:
        content = content.replace(target, replacement)
    engine_path.write_text(content, encoding="utf-8")
    print("Patch applied to engine.py validation")

if __name__ == "__main__":
    main()
