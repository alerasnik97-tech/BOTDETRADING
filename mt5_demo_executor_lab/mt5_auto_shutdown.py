import json
import os
from datetime import datetime, time
import pytz
import MetaTrader5 as mt5

class MT5AutoShutdown:
    def __init__(self, config_path, timeout_manager):
        self.config_path = config_path
        self.timeout_mgr = timeout_manager
        self.config = self._load_config()
        self.tz_ny = pytz.timezone(self.config.get("timezone", "America/New_York"))
        
    def _load_config(self):
        with open(self.config_path, "r") as f:
            return json.load(f)
            
    def get_shutdown_instruction(self):
        """Devuelve la instruccion de cierre basada en la hora de NY y posiciones abiertas"""
        if not self.config.get("auto_shutdown_enabled", False):
            return "CONTINUE_RUNNING"
            
        now_ny = datetime.now(self.tz_ny)
        stop_h, stop_m = map(int, self.config["stop_time_ny"].split(":"))
        stop_time = time(stop_h, stop_m)
        
        # Si aun no es la hora de cierre
        if now_ny.time() < stop_time:
            return "CONTINUE_RUNNING"
            
        # Es hora de cierre o mas tarde. Revisamos posiciones del sistema.
        magic = self.config.get("magic_number", 123456)
        positions = mt5.positions_get(group=f"*{magic}*")
        
        # Filtrado manual por magic si group no es exacto en algunas versiones de MT5
        if positions:
            positions = [p for p in positions if p.magic == magic]
            
        if not positions:
            return "SAFE_TO_SHUTDOWN"
            
        # Hay posiciones. Revisamos si alguna ya cumplio el timeout de 4h.
        # Usamos el timeout_manager para una evaluacion mas profunda si fuera necesario, 
        # pero aqui simplificamos la decision para el orquestador.
        
        # Verificamos si alguna posicion tiene > 4 horas
        now_utc = datetime.now(pytz.UTC)
        for pos in positions:
            open_time_utc = datetime.fromtimestamp(pos.time, tz=pytz.UTC)
            if (now_utc - open_time_utc).total_seconds() >= 4 * 3600:
                return "SHUTDOWN_AFTER_FORCED_DEMO_CLOSE"
                
        # Hay posiciones pero son "jovenes" (< 4h)
        return "SHUTDOWN_DELAYED_POSITION_OPEN"
