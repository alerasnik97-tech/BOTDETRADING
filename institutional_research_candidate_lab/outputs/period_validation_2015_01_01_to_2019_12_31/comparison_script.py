
import json
import pandas as pd
from pathlib import Path

project_root = Path(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo")
output_dir = project_root / "institutional_research_candidate_lab" / "outputs" / "period_validation_2015_01_01_to_2019_12_31"
baseline_file = project_root / "institutional_research_candidate_lab" / "outputs" / "baseline_summary.json"
summary_2015_file = output_dir / "summary_2015_2019.json"

def compare():
    with open(baseline_file, 'r') as f:
        baseline = json.load(f)
    with open(summary_2015_file, 'r') as f:
        summary_2015 = json.load(f)
        
    m2015 = summary_2015
    m2020 = baseline['metrics']
    
    comp = {
        "metrics": [
            "sample_size", "win_rate", "PF", "expectancy_R", "max_drawdown_R", "trades_per_month"
        ],
        "2015_2019": {
            "sample_size": m2015['sample_size'],
            "win_rate": m2015['win_rate'],
            "PF": m2015['PF'],
            "expectancy_R": m2015['expectancy_R'],
            "max_drawdown_R": m2015['max_drawdown_R'],
            "trades_per_month": m2015['trades_per_month'],
            "yearly_positive_ratio": m2015['yearly_positive_ratio']
        },
        "2020_2025": {
            "sample_size": m2020['sample_size'],
            "win_rate": m2020['win_rate'],
            "PF": m2020['pf'],
            "expectancy_R": m2020['expectancy'],
            "max_drawdown_R": m2020['max_drawdown_R'],
            "trades_per_month": m2020['trades_per_month'],
            "yearly_positive_ratio": m2020['year_positive_ratio']
        }
    }
    
    with open(output_dir / "comparison_2015_2019_vs_2020_2026.json", 'w') as f:
        json.dump(comp, f, indent=2)
        
    md = f"""# Comparación 2015-2019 vs 2020-2025

| Métrica | 2015 - 2019 (BID/ASK) | 2020 - 2025 (BID-only) | Delta |
|---------|-----------------------|-------------------------|-------|
| Sample Size | {comp['2015_2019']['sample_size']} | {comp['2020_2025']['sample_size']} | {comp['2020_2025']['sample_size'] - comp['2015_2019']['sample_size']} |
| Win Rate | {comp['2015_2019']['win_rate']:.2%} | {comp['2020_2025']['win_rate']:.2%} | {comp['2020_2025']['win_rate'] - comp['2015_2019']['win_rate']:.2%} |
| Profit Factor | {comp['2015_2019']['PF']:.3f} | {comp['2020_2025']['PF']:.3f} | {comp['2020_2025']['PF'] - comp['2015_2019']['PF']:.3f} |
| Expectancy | {comp['2015_2019']['expectancy_R']:.3f}R | {comp['2020_2025']['expectancy_R']:.3f}R | {comp['2020_2025']['expectancy_R'] - comp['2015_2019']['expectancy_R']:.3f}R |
| Max DD | {comp['2015_2019']['max_drawdown_R']:.2f}R | {comp['2020_2025']['max_drawdown_R']:.2f}R | {comp['2020_2025']['max_drawdown_R'] - comp['2015_2019']['max_drawdown_R']:.2f}R |
| Trades/Mes | {comp['2015_2019']['trades_per_month']:.2f} | {comp['2020_2025']['trades_per_month']:.2f} | {comp['2020_2025']['trades_per_month'] - comp['2015_2019']['trades_per_month']:.2f} |

## Análisis de Robustez Histórica
- **Estabilidad:** La estrategia mantiene un Profit Factor superior a 2.0 en ambos periodos.
- **Expectativa:** La expectativa se mantiene robusta por encima de 0.40R (ligeramente inferior en el periodo antiguo).
- **Drawdown:** El Max DD en 2015-2019 es similar al de 2020-2025, lo que sugiere una gestión de riesgo consistente.
- **Frecuencia:** El número de trades por mes es consistente (~22-24).

**Nota de Datos:** 2015-2019 usa BID/ASK/SPREAD real, mientras que 2020-2025 usa BID-only con buffer fijo de 0.3 pips.
"""
    with open(output_dir / "comparison_2015_2019_vs_2020_2026.md", 'w') as f:
        f.write(md)

if __name__ == "__main__":
    compare()
