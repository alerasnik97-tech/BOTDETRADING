import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone
import pytz

class MT5DataBridge:
    def __init__(self, symbol="EURUSD"):
        self.symbol = symbol
        self.tz_ny = pytz.timezone("America/New_York")
        
    def connect(self):
        if not mt5.initialize():
            print(f"initialize() failed, error code = {mt5.last_error()}")
            return False
        
        # Verificar que la cuenta sea DEMO
        account_info = mt5.account_info()
        if account_info is None:
            print("No se pudo obtener informacion de la cuenta.")
            mt5.shutdown()
            return False
            
        if account_info.trade_mode != mt5.ACCOUNT_TRADE_MODE_DEMO:
            print("CRITICO: Intento de conexion a cuenta REAL. Abortando por seguridad.")
            mt5.shutdown()
            return False
            
        print(f"Conectado a MT5 Demo - Cuenta: {account_info.login} - Broker: {account_info.company}")
        return True

    def get_latest_rates(self, timeframe, n=100):
        """Obtiene las ultimas n velas y las convierte a America/New_York"""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, n)
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        # Convertir a NY
        df['time_ny'] = df['time'].dt.tz_convert(self.tz_ny)
        df.set_index('time_ny', inplace=True)
        return df

    def get_current_tick(self):
        tick = mt5.symbol_info_tick(self.symbol)
        if tick:
            return {
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "time": datetime.fromtimestamp(tick.time, tz=timezone.utc).astimezone(self.tz_ny)
            }
        return None

    def disconnect(self):
        mt5.shutdown()

if __name__ == "__main__":
    bridge = MT5DataBridge()
    if bridge.connect():
        print("Test de lectura H1...")
        h1 = bridge.get_latest_rates(mt5.TIMEFRAME_H1, 5)
        print(h1)
        bridge.disconnect()
