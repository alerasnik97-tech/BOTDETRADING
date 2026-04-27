
import MetaTrader5 as mt5
import json
import os
import sys

def run_connection_check():
    print("--- INICIANDO MT5 CONNECTION CHECK (DEMO ONLY) ---")
    
    config_path = "mt5_local_config.json"
    if not os.path.exists(config_path):
        print("[ERROR] No se encuentra mt5_local_config.json")
        sys.exit(1)
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    if config.get("account_mode") != "DEMO_ONLY" or config.get("allow_live") is True:
        print("[CRITICAL] Configuración NO SEGURA detectada. Abortando.")
        sys.exit(1)
        
    if not mt5.initialize(path=config.get("mt5_terminal_path")):
        print(f"[ERROR] initialize() falló, error code: {mt5.last_error()}")
        sys.exit(1)
        
    authorized = mt5.login(
        login=int(config.get("account_login")),
        password=config.get("password"),
        server=config.get("server")
    )
    
    if authorized:
        acc_info = mt5.account_info()
        print(f"[OK] Conectado a cuenta: {acc_info.login}")
        print(f"[OK] Broker: {acc_info.company}")
        
        # Validar que es DEMO
        if acc_info.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            print("[CRITICAL] ¡CUENTA REAL DETECTADA! Desconectando inmediatamente.")
            mt5.shutdown()
            sys.exit(1)
        else:
            print("[OK] Cuenta verificada como DEMO.")
            
        # Leer spread EURUSD
        symbol = "EURUSD"
        mt5.symbol_select(symbol, True)
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            spread = (tick.ask - tick.bid) / mt5.symbol_info(symbol).point
            print(f"[OK] Spread actual {symbol}: {spread:.1f} points")
        
        print("[SUCCESS] Test de conexión exitoso.")
    else:
        print(f"[ERROR] Fallo al loguear en cuenta {config.get('account_login')}, error code: {mt5.last_error()}")
        
    mt5.shutdown()

if __name__ == "__main__":
    run_connection_check()
