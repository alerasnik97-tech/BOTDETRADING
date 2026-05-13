from __future__ import annotations
from datetime import datetime, time
from zoneinfo import ZoneInfo

class ScheduleGuard:
    """
    Guarda de defensa cronológica intradiaria (Schedule Guard) con soporte nativo
    para ventanas fraccionarias de alta precisión en horas y minutos (D5B Hardening).
    Acota las aperturas a los intervalos operativos de NY y liquida posiciones forzosamente.
    """
    def __init__(
        self,
        entry_start_hour: int | str = 8,
        entry_end_hour: int | str = 11,
        forced_exit_mode: str = "16:00",
        entry_window: str | None = None
    ):
        if entry_window:
            s_str, e_str = entry_window.split("-")
            sh, sm = map(int, s_str.strip().split(":"))
            eh, em = map(int, e_str.strip().split(":"))
            self.start_time = time(sh, sm)
            self.end_time = time(eh, em)
        else:
            if isinstance(entry_start_hour, str) and ":" in entry_start_hour:
                sh, sm = map(int, entry_start_hour.strip().split(":"))
            else:
                sh, sm = int(entry_start_hour), 0
                
            if isinstance(entry_end_hour, str) and ":" in entry_end_hour:
                eh, em = map(int, entry_end_hour.strip().split(":"))
            else:
                eh, em = int(entry_end_hour), 0
                
            self.start_time = time(sh, sm)
            self.end_time = time(eh, em)
            
        self.forced_exit_mode = forced_exit_mode.lower().strip()
        self.forced_exit_time: time | None = None
        
        if self.forced_exit_mode not in ["none", "no_cierre", "false"]:
            val = self.forced_exit_mode
            if ":" in val:
                fh, fm = map(int, val.split(":"))
                self.forced_exit_time = time(fh, fm)
            else:
                digits = "".join(filter(str.isdigit, val))
                if digits:
                    fh = int(digits[:2])
                    fm = int(digits[2:]) if len(digits) > 2 else 0
                    self.forced_exit_time = time(fh, fm)
                else:
                    self.forced_exit_time = time(16, 0)

    def _to_ny(self, timestamp_utc: datetime) -> datetime:
        if timestamp_utc.tzinfo is None:
            ts_utc = timestamp_utc.replace(tzinfo=ZoneInfo("UTC"))
        else:
            ts_utc = timestamp_utc.astimezone(ZoneInfo("UTC"))
        return ts_utc.astimezone(ZoneInfo("America/New_York"))

    def is_entry_permitted(self, timestamp_utc: datetime) -> bool:
        """
        Verifica si la hora actual de NY autoriza el disparo de nuevas entradas
        evaluando con precisión combinada de horas y minutos.
        """
        ny_time = self._to_ny(timestamp_utc)
        return self.start_time <= ny_time.time() <= self.end_time

    def should_force_exit(self, timestamp_utc: datetime) -> bool:
        """
        Verifica si una orden abierta debe ser liquidada inmediatamente
        por el cruce estricto de la marca de tiempo límite en la sesión de NY.
        """
        if self.forced_exit_time is None:
            return False
            
        ny_time = self._to_ny(timestamp_utc)
        return ny_time.time() >= self.forced_exit_time
