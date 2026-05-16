"""UTC canónico; America/New_York derivado con zoneinfo (DST explícito vía IANA)."""
from __future__ import annotations

from datetime import date, datetime, time

import pandas as pd
import zoneinfo

from research_lab.config import NY_TZ


def ny_local_to_utc_iso(local_date: date, hhmm: str) -> tuple[str, str, str]:
    """
    Convierte fecha civil + hora de pared en America/New_York a ISO UTC y NY.

    Returns:
        scheduled_at_utc, scheduled_at_ny, timezone_source
    """
    h, m = (int(x) for x in hhmm.split(":"))
    tz = zoneinfo.ZoneInfo(NY_TZ)
    dt_local = datetime.combine(local_date, time(hour=h, minute=m), tzinfo=tz)
    ts = pd.Timestamp(dt_local)
    utc = ts.tz_convert("UTC")
    ny = ts.tz_convert(NY_TZ)
    return utc.isoformat(), ny.isoformat(), f"iana_{NY_TZ.replace('/', '_').lower()}"
