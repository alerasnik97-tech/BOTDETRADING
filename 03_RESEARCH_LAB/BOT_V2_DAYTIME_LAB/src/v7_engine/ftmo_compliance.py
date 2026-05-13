from __future__ import annotations
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional

class FtmoComplianceEngine:
    """
    Máquina de estados inmutable para garantizar la viabilidad institucional
    bajo las estrictas reglas de capital de FTMO (Sección 3.2).
    """
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.start_of_day_balance = initial_balance
        self.blown = False
        self.blown_reason: str | None = None
        self.current_fx_day_start_utc: datetime | None = None
        self.active_trading_days: set[datetime] = set()

    def update_state(self, timestamp_utc: datetime, closed_pnl: float, floating_pnl: float) -> bool:
        """
        Actualiza el balance incremental, re-evalúa el equity intradiario y determina
        si se ha violado algún umbral crítico de pérdida.
        
        Parámetros:
        - timestamp_utc: Marca temporal actual en UTC.
        - closed_pnl: Ganancia/pérdida realizada incremental que se suma al balance en este instante.
        - floating_pnl: PnL flotante actual de todas las posiciones abiertas.
        
        Retorna: True si la cuenta permanece en cumplimiento, False si entró en Blown State.
        """
        # Si ya está en estado de quiebra, el bloqueo es irreversible
        if self.blown:
            return False
            
        # Sanitizar timestamp a UTC-aware para la conversión precisa a NY
        if timestamp_utc.tzinfo is None:
            ts_utc = timestamp_utc.replace(tzinfo=ZoneInfo("UTC"))
        else:
            ts_utc = timestamp_utc.astimezone(ZoneInfo("UTC"))
            
        ny_time = ts_utc.astimezone(ZoneInfo("America/New_York"))
        
        # Calcular el inicio del día FX actual (17:00 NY del día correspondiente)
        if ny_time.hour >= 17:
            fx_day_ny = ny_time.replace(hour=17, minute=0, second=0, microsecond=0)
        else:
            fx_day_ny = (ny_time - timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)
            
        fx_day_utc = fx_day_ny.astimezone(ZoneInfo("UTC")).replace(tzinfo=None)
        
        # Detectar el cruce de frontera hacia un nuevo día de trading FX
        if self.current_fx_day_start_utc is None or fx_day_utc > self.current_fx_day_start_utc:
            self.current_fx_day_start_utc = fx_day_utc
            self.start_of_day_balance = self.current_balance
            
        # Aplicar el PnL cerrado incremental al balance
        if closed_pnl != 0.0:
            self.current_balance += closed_pnl
            
        current_equity = self.current_balance + floating_pnl
        
        # Registrar actividad si hay transacciones o exposición a riesgo
        if closed_pnl != 0.0 or floating_pnl != 0.0:
            self.active_trading_days.add(self.current_fx_day_start_utc)
            
        # Verificación 1: Límite de Pérdida Diaria (5% del start_of_day_balance)
        min_allowed_daily_equity = self.start_of_day_balance * 0.95
        if current_equity < min_allowed_daily_equity - 1e-5: # Margen de tolerancia de flotante
            self.blown = True
            self.blown_reason = (
                f"Violación de Pérdida Diaria FTMO: Equity ({current_equity:.2f}) "
                f"cayó por debajo del 95% del balance inicial del día ({min_allowed_daily_equity:.2f})"
            )
            return False
            
        # Verificación 2: Límite de Pérdida Absoluta (10% del initial_balance)
        min_allowed_absolute_equity = self.initial_balance * 0.90
        if current_equity < min_allowed_absolute_equity - 1e-5:
            self.blown = True
            self.blown_reason = (
                f"Violación de Pérdida Absoluta FTMO: Equity ({current_equity:.2f}) "
                f"cayó por debajo del 90% del capital inicial ({min_allowed_absolute_equity:.2f})"
            )
            return False
            
        return True

    def get_position_risk_amount(self) -> float:
        """
        Retorna el monto exacto en dólares a arriesgar para la siguiente posición,
        fijado incondicionalmente al 1% del balance actual.
        """
        return self.current_balance * 0.01

    def get_minimum_days_count(self) -> int:
        """
        Retorna la cantidad de días únicos de trading FX con actividad registrada.
        """
        return len(self.active_trading_days)
