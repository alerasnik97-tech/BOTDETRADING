import MetaTrader5 as mt5

class MT5KillSwitch:
    def __init__(self, max_consecutive_losses=3, max_drawdown=0.05):
        self.max_consecutive_losses = max_consecutive_losses
        self.max_drawdown = max_drawdown
        self.is_active = False
        
    def check_conditions(self, initial_balance):
        account_info = mt5.account_info()
        if not account_info:
            return True # Detener por falta de info
            
        current_balance = account_info.balance
        drawdown = (initial_balance - current_balance) / initial_balance
        
        if drawdown >= self.max_drawdown:
            print(f"KILL SWITCH: Drawdown critico detectado ({drawdown:.2%}). DETENCION.")
            self.is_active = True
            return True
            
        return False
        
    def check_spread(self, symbol, max_spread_pips=5.0):
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return True
            
        spread = (tick.ask - tick.bid) / 0.0001 # Para EURUSD
        if spread >= max_spread_pips:
            print(f"KILL SWITCH: Spread anormal detectado ({spread:.1f} pips).")
            return True
        return False
