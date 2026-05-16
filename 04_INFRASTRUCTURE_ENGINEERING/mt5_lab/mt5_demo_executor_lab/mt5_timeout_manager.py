import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

class MT5TimeoutManager:
    def __init__(self, timeout_hours=4):
        self.timeout_hours = timeout_hours
        
    def check_timeouts(self, router):
        """Monitorea posiciones y cierra las que superen el timeout"""
        positions = mt5.positions_get(symbol=router.symbol)
        if not positions:
            return
            
        now = datetime.now(timezone.utc)
        
        for pos in positions:
            # Los tiempos en MT5 suelen ser UTC o Server Time. 
            # Convertimos el tiempo de creacion a datetime con zona horaria
            open_time = datetime.fromtimestamp(pos.time, tz=timezone.utc)
            elapsed = now - open_time
            
            if elapsed >= timedelta(hours=self.timeout_hours):
                print(f"TIMEOUT: Cerrando posicion {pos.ticket} tras {elapsed.total_seconds()/3600:.2f} horas.")
                router.close_position(pos.ticket, comment="Timeout 4h")
