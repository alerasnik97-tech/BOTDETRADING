import os
from shadow_line_lab import config

def generate_shadow_report(summary_data):
    report_md = f"""# Shadow Line Run Report
## Veredicto Ejecutivo

**Línea:** `{config.STRATEGY_CONFIG['variant_id']}`
**Fecha de Reporte:** `{summary_data.get('timestamp', 'N/A')}`
**Estado:** `SHADOW_ONLY`

---

## Resumen de la Corrida
- trades_executed: {summary_data.get('trades_executed', 0)}
- total_pnl_r: {summary_data.get('total_pnl_r', 0.0)}
- win_rate: {summary_data.get('win_rate', 0.0)}%
- news_blocked_count: {summary_data.get('news_blocked_count', 0)}

---

## Observaciones Técnicas
Este reporte fue generado de forma automática por la infraestructura aislada de la Shadow Line. No tiene impacto sobre los ledgers oficiales de producción.
"""
    os.makedirs(os.path.dirname(config.REPORT_FILE), exist_ok=True)
    with open(config.REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report_md)
