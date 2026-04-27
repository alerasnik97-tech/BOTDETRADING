"""
Connector BLS CPI/PPI usando requests con headers de navegador.
Extrae fechas oficiales de las páginas schedule de BLS.
"""
from __future__ import annotations

import re
import json
import urllib.request
import urllib.error
import ssl
from datetime import date, datetime
from pathlib import Path
from typing import Any


def _fetch_bls_page(url: str) -> str | None:
    """Intenta obtener contenido de página BLS con headers de navegador."""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url, headers=headers)
    
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            import gzip
            if response.headers.get('Content-Encoding') == 'gzip':
                return gzip.decompress(response.read()).decode('utf-8', errors='ignore')
            return response.read().decode('utf-8', errors='ignore')
    except urllib.error.HTTPError as e:
        return f"HTTP_ERROR:{e.code}:{e.reason}"
    except Exception as e:
        return f"ERROR:{type(e).__name__}:{str(e)}"


def parse_cpi_from_schedule(html: str, year: int) -> list[dict]:
    """Parsea fechas CPI desde HTML de schedule BLS."""
    events = []
    
    # Patrones comunes en páginas BLS schedule
    # Ejemplo: "Jan 11" o "January 11" o "Jan 11, 2024"
    cpi_patterns = [
        r'Consumer\s+Price\s+Index.*?([A-Za-z]+)\s+(\d{1,2})[\s,]*' + str(year),
        r'CPI.*?([A-Za-z]+)\s+(\d{1,2})[\s,]*' + str(year),
        r'([A-Za-z]+)\s+(\d{1,2})[\s,]*' + str(year) + r'.*?Consumer\s+Price',
    ]
    
    month_map = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }
    
    for pattern in cpi_patterns:
        matches = re.finditer(pattern, html, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                month_str = match.group(1).lower()
                day = int(match.group(2))
                month = month_map.get(month_str)
                
                if month and 1 <= day <= 31:
                    event_date = date(year, month, day)
                    events.append({
                        'title': 'cpi m/m',
                        'country': 'United States',
                        'currency': 'USD',
                        'local_date_ny': event_date.isoformat(),
                        'local_time_ny': '08:30',
                        'source': f'bls_cpi_schedule_{year}',
                        'source_type': 'official_web',
                        'source_url': f'https://www.bls.gov/schedule/{year}/home.htm',
                        'anchor_group': 'CPI',
                        'importance': 'HIGH',
                        'notes': f'Parsed from BLS schedule {year}',
                    })
            except (ValueError, AttributeError):
                continue
    
    return events


def parse_ppi_from_schedule(html: str, year: int) -> list[dict]:
    """Parsea fechas PPI desde HTML de schedule BLS."""
    events = []
    
    ppi_patterns = [
        r'Producer\s+Price\s+Index.*?([A-Za-z]+)\s+(\d{1,2})[\s,]*' + str(year),
        r'PPI.*?([A-Za-z]+)\s+(\d{1,2})[\s,]*' + str(year),
        r'([A-Za-z]+)\s+(\d{1,2})[\s,]*' + str(year) + r'.*?Producer\s+Price',
    ]
    
    month_map = {
        'jan': 1, 'january': 1,
        'feb': 2, 'february': 2,
        'mar': 3, 'march': 3,
        'apr': 4, 'april': 4,
        'may': 5,
        'jun': 6, 'june': 6,
        'jul': 7, 'july': 7,
        'aug': 8, 'august': 8,
        'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10,
        'nov': 11, 'november': 11,
        'dec': 12, 'december': 12,
    }
    
    for pattern in ppi_patterns:
        matches = re.finditer(pattern, html, re.IGNORECASE | re.DOTALL)
        for match in matches:
            try:
                month_str = match.group(1).lower()
                day = int(match.group(2))
                month = month_map.get(month_str)
                
                if month and 1 <= day <= 31:
                    event_date = date(year, month, day)
                    events.append({
                        'title': 'ppi m/m',
                        'country': 'United States',
                        'currency': 'USD',
                        'local_date_ny': event_date.isoformat(),
                        'local_time_ny': '08:30',
                        'source': f'bls_ppi_schedule_{year}',
                        'source_type': 'official_web',
                        'source_url': f'https://www.bls.gov/schedule/{year}/home.htm',
                        'anchor_group': 'PPI',
                        'importance': 'HIGH',
                        'notes': f'Parsed from BLS schedule {year}',
                    })
            except (ValueError, AttributeError):
                continue
    
    return events


def fetch_bls_cpi_ppi_events(years: list[int] = None) -> dict:
    """
    Fetches CPI and PPI events from BLS official schedule pages.
    Returns ConnectorResult-style dict.
    """
    if years is None:
        years = [2024, 2025, 2026]
    
    all_events = []
    access_log = []
    total_fetched = 0
    
    for year in years:
        url = f"https://www.bls.gov/schedule/{year}/home.htm"
        result = _fetch_bls_page(url)
        
        if result and not result.startswith(("HTTP_ERROR", "ERROR")):
            # Guardar respuesta para debug
            debug_path = Path(f"bls_response_{year}.html")
            try:
                debug_path.write_text(result[:50000], encoding="utf-8")  # Limitar tamaño
            except:
                pass
            
            cpi_events = parse_cpi_from_schedule(result, year)
            ppi_events = parse_ppi_from_schedule(result, year)
            
            all_events.extend(cpi_events)
            all_events.extend(ppi_events)
            
            access_log.append({
                "year": year,
                "url": url,
                "status": "success",
                "bytes": len(result),
                "cpi_found": len(cpi_events),
                "ppi_found": len(ppi_events),
            })
            total_fetched += len(cpi_events) + len(ppi_events)
        else:
            access_log.append({
                "year": year,
                "url": url,
                "status": "error",
                "error": result,
                "cpi_found": 0,
                "ppi_found": 0,
            })
    
    # Intentar URLs alternativas si las principales fallaron
    if total_fetched == 0:
        alt_urls = [
            "https://www.bls.gov/schedule/news_release/cpi.htm",
            "https://www.bls.gov/schedule/news_release/ppi.htm",
        ]
        for url in alt_urls:
            result = _fetch_bls_page(url)
            access_log.append({
                "url": url,
                "status": "attempted_alternative",
                "result": result[:200] if result else None,
            })
    
    return {
        "connector_id": "bls_cpi_ppi_auto",
        "events": all_events,
        "status": "ok" if all_events else "blocked",
        "message": f"Fetched {len(all_events)} events from BLS" if all_events else "BLS pages blocked or no events found",
        "meta": {
            "years_attempted": years,
            "access_log": access_log,
        },
    }


if __name__ == "__main__":
    # Test del conector
    result = fetch_bls_cpi_ppi_events([2024, 2025, 2026])
    print(json.dumps(result, indent=2, default=str))
