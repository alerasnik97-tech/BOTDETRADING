"""
Connector Híbrido Multi-Capa BLS CPI/PPI
Intenta múltiples superficies oficiales BLS con múltiples métodos de adquisición.
"""
from __future__ import annotations

import re
import json
import urllib.request
import urllib.error
import http.client
import ssl
import subprocess
import sys
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class BLSScheduleParser(HTMLParser):
    """Parser HTML para extraer eventos de calendario BLS."""
    
    def __init__(self, year: int):
        super().__init__()
        self.year = year
        self.events = []
        self.in_table = False
        self.in_row = False
        self.in_cell = False
        self.current_cell_text = ""
        self.current_row = []
        self.month_map = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
        }
    
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag in ['table', 'tbody']:
            self.in_table = True
        elif tag == 'tr' and self.in_table:
            self.in_row = True
            self.current_row = []
        elif tag in ['td', 'th'] and self.in_row:
            self.in_cell = True
            self.current_cell_text = ""
    
    def handle_endtag(self, tag):
        if tag in ['td', 'th'] and self.in_cell:
            self.in_cell = False
            self.current_row.append(self.current_cell_text.strip())
        elif tag == 'tr' and self.in_row:
            self.in_row = False
            self._process_row()
        elif tag in ['table', 'tbody']:
            self.in_table = False
    
    def handle_data(self, data):
        if self.in_cell:
            self.current_cell_text += data
    
    def _process_row(self):
        """Procesa una fila de tabla buscando CPI/PPI."""
        if not self.current_row:
            return
        
        row_text = ' '.join(self.current_row).lower()
        
        # Detectar CPI
        if any(k in row_text for k in ['consumer price index', 'cpi', 'consumer price']):
            date_match = self._extract_date_from_row()
            if date_match:
                self.events.append({
                    'title': 'cpi m/m',
                    'country': 'United States',
                    'currency': 'USD',
                    'local_date_ny': date_match,
                    'local_time_ny': '08:30',
                    'source': f'bls_cpi_schedule_{self.year}',
                    'source_type': 'official_web',
                    'source_url': f'https://www.bls.gov/schedule/{self.year}/home.htm',
                    'anchor_group': 'CPI',
                    'importance': 'HIGH',
                    'notes': f'Parsed from BLS schedule table {self.year}',
                })
        
        # Detectar PPI
        if any(k in row_text for k in ['producer price index', 'ppi', 'producer price']):
            date_match = self._extract_date_from_row()
            if date_match:
                self.events.append({
                    'title': 'ppi m/m',
                    'country': 'United States',
                    'currency': 'USD',
                    'local_date_ny': date_match,
                    'local_time_ny': '08:30',
                    'source': f'bls_ppi_schedule_{self.year}',
                    'source_type': 'official_web',
                    'source_url': f'https://www.bls.gov/schedule/{self.year}/home.htm',
                    'anchor_group': 'PPI',
                    'importance': 'HIGH',
                    'notes': f'Parsed from BLS schedule table {self.year}',
                })
    
    def _extract_date_from_row(self) -> str | None:
        """Extrae fecha del texto de la fila."""
        row_text = ' '.join(self.current_row)
        
        # Patrones de fecha: "Jan 11", "January 11", "11 Jan", "11 January", "1/11/2024", etc.
        patterns = [
            r'([A-Za-z]{3,9})\s+(\d{1,2})[,\s]+' + str(self.year),
            r'(\d{1,2})\s+([A-Za-z]{3,9})[,\s]+' + str(self.year),
            r'(\d{1,2})/(\d{1,2})/' + str(self.year),
            r'(\d{1,2})-(\d{1,2})-' + str(self.year),
        ]
        
        for pattern in patterns:
            match = re.search(pattern, row_text, re.IGNORECASE)
            if match:
                try:
                    if '/' in pattern or '-' in pattern:
                        # MM/DD/YYYY o MM-DD-YYYY
                        month, day = int(match.group(1)), int(match.group(2))
                    else:
                        # Mes nombre + día
                        part1, part2 = match.group(1).lower(), match.group(2)
                        if part1.isdigit():
                            day, month_str = int(part1), part2
                            month = self.month_map.get(month_str)
                        else:
                            month_str, day = part1, int(part2)
                            month = self.month_map.get(month_str)
                    
                    if month and 1 <= day <= 31:
                        return date(self.year, month, day).isoformat()
                except (ValueError, TypeError):
                    continue
        return None


def fetch_with_urllib(url: str, method_name: str = "urllib") -> tuple[str | None, str]:
    """Intenta fetch con urllib usando diferentes estrategias de headers."""
    
    # Estrategia 1: Headers mínimos de navegador moderno
    headers_sets = [
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.bls.gov/",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Cache-Control": "max-age=0",
        },
        {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        },
    ]
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    for i, headers in enumerate(headers_sets):
        try:
            req = urllib.request.Request(url, headers=headers, method='GET')
            with urllib.request.urlopen(req, context=ctx, timeout=45) as response:
                data = response.read()
                
                # Manejar compresión
                content_encoding = response.headers.get('Content-Encoding', '')
                if 'gzip' in content_encoding:
                    import gzip
                    data = gzip.decompress(data)
                elif 'deflate' in content_encoding:
                    import zlib
                    data = zlib.decompress(data)
                
                return data.decode('utf-8', errors='ignore'), f"{method_name}_headers_{i+1}"
        except urllib.error.HTTPError as e:
            if i == len(headers_sets) - 1:
                return None, f"HTTP_{e.code}"
            continue
        except Exception as e:
            if i == len(headers_sets) - 1:
                return None, f"ERROR_{type(e).__name__}"
            continue
    
    return None, "FAILED_ALL_HEADERS"


def fetch_with_httpclient(url: str) -> tuple[str | None, str]:
    """Intenta fetch con http.client directo."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        if parsed.scheme == 'https':
            conn = http.client.HTTPSConnection(parsed.netloc, context=ctx, timeout=45)
        else:
            conn = http.client.HTTPConnection(parsed.netloc, timeout=45)
        
        headers = {
            "Host": parsed.netloc,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }
        
        path = parsed.path or '/'
        if parsed.query:
            path += '?' + parsed.query
        
        conn.request("GET", path, headers=headers)
        response = conn.getresponse()
        
        if response.status == 200:
            data = response.read()
            return data.decode('utf-8', errors='ignore'), "httpclient"
        else:
            return None, f"HTTP_{response.status}"
    except Exception as e:
        return None, f"ERROR_{type(e).__name__}"
    finally:
        try:
            conn.close()
        except:
            pass


def fetch_with_curl(url: str) -> tuple[str | None, str]:
    """Intenta fetch con curl via subprocess."""
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', '-A', 
             'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
             '--max-time', '30', '-k', url],
            capture_output=True,
            text=True,
            timeout=35
        )
        if result.returncode == 0 and result.stdout:
            return result.stdout, "curl"
        return None, f"CURL_EXIT_{result.returncode}"
    except FileNotFoundError:
        return None, "CURL_NOT_FOUND"
    except Exception as e:
        return None, f"CURL_ERROR_{type(e).__name__}"


def fetch_bls_page_multi_method(url: str) -> tuple[str | None, dict]:
    """Intenta fetch con múltiples métodos hasta conseguir contenido."""
    methods_tried = []
    
    # Método 1: urllib con headers variados
    content, method = fetch_with_urllib(url, "urllib")
    methods_tried.append({"method": "urllib", "result": "success" if content else method})
    if content:
        return content, {"success": True, "method": "urllib", "methods_tried": methods_tried}
    
    # Método 2: http.client
    content, method = fetch_with_httpclient(url)
    methods_tried.append({"method": "httpclient", "result": "success" if content else method})
    if content:
        return content, {"success": True, "method": "httpclient", "methods_tried": methods_tried}
    
    # Método 3: curl subprocess
    content, method = fetch_with_curl(url)
    methods_tried.append({"method": "curl", "result": "success" if content else method})
    if content:
        return content, {"success": True, "method": "curl", "methods_tried": methods_tried}
    
    return None, {"success": False, "methods_tried": methods_tried}


def parse_events_from_html(html: str, year: int, source_url: str) -> list[dict]:
    """Parsea eventos CPI/PPI desde HTML usando múltiples estrategias."""
    events = []
    
    if not html or len(html) < 100:
        return events
    
    # Estrategia 1: Parser HTML estructurado
    parser = BLSScheduleParser(year)
    try:
        parser.feed(html)
        events.extend(parser.events)
    except Exception:
        pass
    
    # Estrategia 2: Regex directo si el parser no encontró nada
    if not events:
        events.extend(_parse_with_regex(html, year, source_url))
    
    # Deduplicar por fecha + tipo
    seen = set()
    unique_events = []
    for ev in events:
        key = (ev['anchor_group'], ev['local_date_ny'])
        if key not in seen:
            seen.add(key)
            unique_events.append(ev)
    
    return unique_events


def _parse_with_regex(html: str, year: int, source_url: str) -> list[dict]:
    """Parseo fallback con regex."""
    events = []
    month_map = {
        'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
        'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
        'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
        'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12,
    }
    
    # Buscar Consumer Price Index con fecha
    cpi_patterns = [
        r'Consumer\s+Price\s+Index[^\n]*?(?:for\s+([A-Za-z]+)\s+\d+)?[^\n]*?(\d{1,2})[\s,]+([A-Za-z]{3,9})[\s,]+' + str(year),
        r'Consumer\s+Price\s+Index[^\n]*?(\d{1,2})/(\d{1,2})/' + str(year),
        r'CPI[^\n]{0,50}(\d{1,2})[\s,]+([A-Za-z]{3,9})[\s,]+' + str(year),
    ]
    
    for pattern in cpi_patterns:
        for match in re.finditer(pattern, html, re.IGNORECASE):
            try:
                groups = match.groups()
                if len(groups) >= 2:
                    if groups[0].isdigit() and groups[1].isdigit():
                        month, day = int(groups[0]), int(groups[1])
                    else:
                        day_str = groups[0] if groups[0].isdigit() else groups[-2]
                        month_str = groups[1].lower() if not groups[1].isdigit() else groups[-1].lower()
                        day = int(day_str) if day_str.isdigit() else int(groups[-2])
                        month = month_map.get(month_str)
                    
                    if month and 1 <= day <= 31:
                        event_date = date(year, month, day)
                        events.append({
                            'title': 'cpi m/m',
                            'country': 'United States',
                            'currency': 'USD',
                            'local_date_ny': event_date.isoformat(),
                            'local_time_ny': '08:30',
                            'source': f'bls_cpi_{year}',
                            'source_type': 'official_web',
                            'source_url': source_url,
                            'anchor_group': 'CPI',
                            'importance': 'HIGH',
                            'notes': f'Regex parsed from {source_url}',
                        })
            except Exception:
                continue
    
    # Similar para PPI
    ppi_patterns = [
        r'Producer\s+Price\s+Index[^\n]*?(?:for\s+([A-Za-z]+)\s+\d+)?[^\n]*?(\d{1,2})[\s,]+([A-Za-z]{3,9})[\s,]+' + str(year),
        r'Producer\s+Price\s+Index[^\n]*?(\d{1,2})/(\d{1,2})/' + str(year),
        r'PPI[^\n]{0,50}(\d{1,2})[\s,]+([A-Za-z]{3,9})[\s,]+' + str(year),
    ]
    
    for pattern in ppi_patterns:
        for match in re.finditer(pattern, html, re.IGNORECASE):
            try:
                groups = match.groups()
                if len(groups) >= 2:
                    if groups[0].isdigit() and groups[1].isdigit():
                        month, day = int(groups[0]), int(groups[1])
                    else:
                        day_str = groups[0] if groups[0].isdigit() else groups[-2]
                        month_str = groups[1].lower() if not groups[1].isdigit() else groups[-1].lower()
                        day = int(day_str) if day_str.isdigit() else int(groups[-2])
                        month = month_map.get(month_str)
                    
                    if month and 1 <= day <= 31:
                        event_date = date(year, month, day)
                        events.append({
                            'title': 'ppi m/m',
                            'country': 'United States',
                            'currency': 'USD',
                            'local_date_ny': event_date.isoformat(),
                            'local_time_ny': '08:30',
                            'source': f'bls_ppi_{year}',
                            'source_type': 'official_web',
                            'source_url': source_url,
                            'anchor_group': 'PPI',
                            'importance': 'HIGH',
                            'notes': f'Regex parsed from {source_url}',
                        })
            except Exception:
                continue
    
    return events


def fetch_bls_cpi_ppi_hybrid(years: list[int] = None) -> dict:
    """
    Fetches CPI/PPI usando estrategia híbrida multi-capa.
    Capa 1: Páginas anuales
    Capa 2: Páginas específicas
    Capa 3: Archived releases
    """
    if years is None:
        years = [2024, 2025, 2026]
    
    all_events = []
    layer_log = []
    
    # CAPA 1: Páginas anuales de calendario
    for year in years:
        url = f"https://www.bls.gov/schedule/{year}/home.htm"
        content, meta = fetch_bls_page_multi_method(url)
        
        layer_log.append({
            "layer": 1,
            "type": "annual_schedule",
            "year": year,
            "url": url,
            "success": meta["success"],
            "method_used": meta.get("method", "none"),
            "methods_tried": [m["method"] for m in meta.get("methods_tried", [])],
        })
        
        if content:
            events = parse_events_from_html(content, year, url)
            all_events.extend(events)
    
    # CAPA 2: Páginas específicas de release (si no tenemos suficientes eventos)
    if len([e for e in all_events if e['anchor_group'] == 'CPI']) < len(years) * 6:
        specific_urls = [
            ("https://www.bls.gov/schedule/news_release/cpi.htm", "CPI"),
            ("https://www.bls.gov/schedule/news_release/ppi.htm", "PPI"),
        ]
        for url, event_type in specific_urls:
            content, meta = fetch_bls_page_multi_method(url)
            layer_log.append({
                "layer": 2,
                "type": f"specific_{event_type.lower()}",
                "url": url,
                "success": meta["success"],
                "method_used": meta.get("method", "none"),
            })
            if content:
                for year in years:
                    events = parse_events_from_html(content, year, url)
                    all_events.extend(events)
    
    # CAPA 3: Archived releases (respaldo)
    archived_urls = [
        "https://www.bls.gov/bls/news-release/cpi.htm",
        "https://www.bls.gov/bls/news-release/ppi.htm",
    ]
    for url in archived_urls:
        content, meta = fetch_bls_page_multi_method(url)
        layer_log.append({
            "layer": 3,
            "type": "archived",
            "url": url,
            "success": meta["success"],
            "method_used": meta.get("method", "none"),
        })
        if content:
            for year in years:
                events = parse_events_from_html(content, year, url)
                all_events.extend(events)
    
    # Deduplicar final
    seen = set()
    unique_events = []
    for ev in all_events:
        key = (ev['anchor_group'], ev['local_date_ny'])
        if key not in seen:
            seen.add(key)
            unique_events.append(ev)
    
    cpi_count = len([e for e in unique_events if e['anchor_group'] == 'CPI'])
    ppi_count = len([e for e in unique_events if e['anchor_group'] == 'PPI'])
    
    return {
        "connector_id": "bls_cpi_ppi_hybrid",
        "events": unique_events,
        "status": "ok" if unique_events else "blocked",
        "message": f"Fetched {cpi_count} CPI and {ppi_count} PPI events" if unique_events else "All BLS layers blocked or no events found",
        "meta": {
            "years_attempted": years,
            "layer_log": layer_log,
            "cpi_count": cpi_count,
            "ppi_count": ppi_count,
        },
    }


if __name__ == "__main__":
    result = fetch_bls_cpi_ppi_hybrid([2024, 2025, 2026])
    print(json.dumps(result, indent=2, default=str))
