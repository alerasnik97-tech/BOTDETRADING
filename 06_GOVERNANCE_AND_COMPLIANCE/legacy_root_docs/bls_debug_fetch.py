"""
Debug script para guardar HTML de BLS y analizar estructura.
"""
import subprocess
from pathlib import Path

urls = [
    "https://www.bls.gov/schedule/2024/home.htm",
    "https://www.bls.gov/schedule/2025/home.htm", 
    "https://www.bls.gov/schedule/2026/home.htm",
    "https://www.bls.gov/schedule/news_release/cpi.htm",
    "https://www.bls.gov/schedule/news_release/ppi.htm",
]

output_dir = Path("bls_html_samples")
output_dir.mkdir(exist_ok=True)

for url in urls:
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', '-A', 
             'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
             '--max-time', '30', '-k', url],
            capture_output=True,
            text=True,
            timeout=35,
            encoding='utf-8',
            errors='ignore'
        )
        if result.returncode == 0 and result.stdout:
            filename = url.replace('https://', '').replace('/', '_') + '.html'
            filepath = output_dir / filename
            filepath.write_text(result.stdout[:100000], encoding='utf-8')
            print(f"Saved: {filename} ({len(result.stdout)} bytes)")
            
            # Buscar menciones de CPI/PPI en el contenido
            content_lower = result.stdout.lower()
            cpi_mentions = content_lower.count('consumer price index')
            ppi_mentions = content_lower.count('producer price index')
            print(f"  CPI mentions: {cpi_mentions}, PPI mentions: {ppi_mentions}")
            
            # Mostrar primeros 500 chars
            print(f"  Preview: {result.stdout[:500]}")
            print()
    except Exception as e:
        print(f"Error fetching {url}: {e}")

print("\nDone. Check bls_html_samples/ directory.")
