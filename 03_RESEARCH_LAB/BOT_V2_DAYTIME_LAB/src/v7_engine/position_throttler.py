from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

class PositionThrottler:
    """
    Controlador de flujo estricto (Position Throttler) para limitar el volumen intradiario
    a un máximo pre-comprometido de operaciones por día FX (Sección 3.3).
    Incorpora registro de auditoría pormenorizado de rechazos (D5B Hardening).
    """
    def __init__(self, max_trades_per_day: int = 3):
        self.max_trades = max_trades_per_day
        self.current_day_trades = 0
        self.current_day_rejected = 0
        self.current_fx_day_start_utc: datetime | None = None
        self.rejection_log: list[dict] = []

    def _update_fx_day(self, timestamp_utc: datetime) -> None:
        """
        Evalúa y actualiza la frontera del día FX (17:00 NY), reiniciando los contadores
        al detectarse el cruce cronológico.
        """
        if timestamp_utc.tzinfo is None:
            ts_utc = timestamp_utc.replace(tzinfo=ZoneInfo("UTC"))
        else:
            ts_utc = timestamp_utc.astimezone(ZoneInfo("UTC"))
            
        ny_time = ts_utc.astimezone(ZoneInfo("America/New_York"))
        
        if ny_time.hour >= 17:
            fx_day_ny = ny_time.replace(hour=17, minute=0, second=0, microsecond=0)
        else:
            fx_day_ny = (ny_time - timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
            
        fx_day_utc = fx_day_ny.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        
        if self.current_fx_day_start_utc is None or fx_day_utc > self.current_fx_day_start_utc:
            self.current_fx_day_start_utc = fx_day_utc
            self.current_day_trades = 0
            self.current_day_rejected = 0

    def can_trade(self, timestamp_utc: datetime) -> bool:
        """
        Consulta si queda cuota disponible en el día FX actual sin consumir el token.
        """
        self._update_fx_day(timestamp_utc)
        return self.current_day_trades < self.max_trades

    def allow_trade(self, timestamp_utc: datetime) -> bool:
        """
        Puerta de decisión FIFO. Evalúa y de estar autorizada, consume una cuota de operación.
        Si el cupo está agotado, incrementa el registro de rechazos y asienta los detalles forenses.
        
        Retorna: True si se autoriza la entrada, False si se estrangula.
        """
        self._update_fx_day(timestamp_utc)
        
        if self.current_day_trades < self.max_trades:
            self.current_day_trades += 1
            return True
        else:
            self.current_day_rejected += 1
            self.rejection_log.append({
                "timestamp": timestamp_utc.isoformat(),
                "fx_day_start": self.current_fx_day_start_utc.isoformat() if self.current_fx_day_start_utc else None,
                "reason": f"QUOTA_EXCEEDED_MAX_{self.max_trades}",
                "rejected_signal_number": self.current_day_rejected
            })
            return False

    def get_rejected_count(self) -> int:
        """Retorna la cantidad de señales estranguladas en la jornada FX en curso."""
        return self.current_day_rejected

    def get_rejection_log(self) -> list[dict]:
        """Retorna el libro mayor de auditoría con el detalle de todas las señales bloqueadas."""
        return self.rejection_log
