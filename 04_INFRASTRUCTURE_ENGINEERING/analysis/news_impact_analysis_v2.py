"""
NEWS IMPACT ANALYSIS v2
Cruza dataset official_anchors (190 eventos) con trades 2024-2025.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


def parse_datetime(dt_str: str) -> datetime:
    """Parsea datetime en formato '2024-01-24 05:00:00'."""
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")


def load_anchor_events() -> list[dict]:
    """Carga eventos desde canonical_anchor_events.csv."""
    csv_path = Path("c:/Users/alera/Desktop/BOT DE TRADING CURSOR/data/official_anchors/out/canonical_anchor_events.csv")
    events = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ny_time_str = row.get('scheduled_at_ny', '')
            if ny_time_str:
                try:
                    base_time = ny_time_str[:19]
                    dt = datetime.strptime(base_time, "%Y-%m-%dT%H:%M:%S")
                    events.append({
                        'event_id': row['event_id'],
                        'title': row['title'],
                        'anchor_group': row['anchor_group'],
                        'scheduled_at_ny': dt,
                        'currency': row['currency'],
                        'importance': row['importance'],
                    })
                except Exception:
                    continue
    return events


def load_trades() -> list[dict]:
    """Carga trades desde trades_consolidated_2020_2025.csv (rango completo 2020-2025)."""
    csv_path = Path("c:/Users/alera/Desktop/BOT DE TRADING CURSOR/trades_consolidated_2020_2025.csv")
    trades = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                entry_time = parse_datetime(row['entry_time_ny'])
                # Incluir todos los años disponibles en el dataset consolidado
                if entry_time.year not in [2020, 2021, 2022, 2023, 2024, 2025]:
                    continue
                    
                trades.append({
                    'entry_time': entry_time,
                    'pnl_r': float(row['pnl_r']),
                    'result': row['result'],
                    'exit_reason': row['exit_reason'],
                })
            except Exception:
                continue
    return trades


def time_to_event_minutes(trade_time: datetime, event_time: datetime) -> int:
    """Minutos desde el trade hasta el evento (negativo = antes del evento)."""
    diff = (trade_time - event_time).total_seconds() / 60
    return int(diff)


def analyze_impact(trades: list[dict], events: list[dict]) -> dict:
    """Analiza impacto de eventos sobre trades."""
    
    # Agrupar eventos por anchor_group
    events_by_group = defaultdict(list)
    for ev in events:
        events_by_group[ev['anchor_group']].append(ev)
    
    windows = [5, 15, 30, 60]
    
    results = {
        'summary': {
            'total_trades': len(trades),
            'total_events': len(events),
            'trades_by_year': defaultdict(int),
        },
        'by_anchor_group': {},
        'by_window': {},
        'closest_trades': [],
    }
    
    # Contar trades por año
    for trade in trades:
        results['summary']['trades_by_year'][trade['entry_time'].year] += 1
    
    # Métricas por anchor_group
    for group, group_events in events_by_group.items():
        months_dist = defaultdict(int)
        hours_dist = defaultdict(int)
        for ev in group_events:
            months_dist[ev['scheduled_at_ny'].strftime("%Y-%m")] += 1
            hours_dist[ev['scheduled_at_ny'].hour] += 1
            
        results['by_anchor_group'][group] = {
            'total_events': len(group_events),
            'years': sorted(set(ev['scheduled_at_ny'].year for ev in group_events)),
            'months_distribution': dict(sorted(months_dist.items())),
            'hours_distribution': dict(sorted(hours_dist.items())),
        }
    
    # Baseline
    baseline_wins = sum(1 for t in trades if t['result'] == 'win')
    baseline_wr = baseline_wins / len(trades) * 100 if trades else 0
    baseline_pnl = sum(t['pnl_r'] for t in trades) / len(trades) if trades else 0
    
    results['baseline'] = {
        'win_rate': baseline_wr,
        'avg_pnl_r': baseline_pnl,
        'total_trades': len(trades),
    }
    
    # Análisis por ventana
    for window in windows:
        trades_affected = []
        
        for trade in trades:
            entry = trade['entry_time']
            closest_event = None
            min_dist = float('inf')
            closest_group = None
            
            for group, group_events in events_by_group.items():
                for ev in group_events:
                    dist = abs((entry - ev['scheduled_at_ny']).total_seconds() / 60)
                    if dist < min_dist:
                        min_dist = dist
                        closest_event = ev
                        closest_group = group
            
            if min_dist <= window:
                trades_affected.append({
                    **trade,
                    'distance_min': min_dist,
                    'anchor_group': closest_group,
                    'event_time': closest_event['scheduled_at_ny'].isoformat() if closest_event else None,
                })
        
        # Stats
        if trades_affected:
            wins = sum(1 for t in trades_affected if t['result'] == 'win')
            wr = wins / len(trades_affected) * 100
            avg_pnl = sum(t['pnl_r'] for t in trades_affected) / len(trades_affected)
        else:
            wr = 0
            avg_pnl = 0
        
        # Por grupo
        by_group = defaultdict(lambda: {'count': 0, 'wins': 0, 'pnl_sum': 0})
        for t in trades_affected:
            g = t['anchor_group']
            by_group[g]['count'] += 1
            if t['result'] == 'win':
                by_group[g]['wins'] += 1
            by_group[g]['pnl_sum'] += t['pnl_r']
        
        results['by_window'][window] = {
            'trades_count': len(trades_affected),
            'percentage_of_total': len(trades_affected) / len(trades) * 100 if trades else 0,
            'win_rate': wr,
            'avg_pnl_r': avg_pnl,
            'delta_win_rate': wr - baseline_wr,
            'delta_pnl': avg_pnl - baseline_pnl,
            'by_group': {
                g: {
                    'count': v['count'],
                    'win_rate': v['wins'] / v['count'] * 100 if v['count'] else 0,
                    'avg_pnl': v['pnl_sum'] / v['count'] if v['count'] else 0,
                }
                for g, v in by_group.items()
            },
        }
    
    # Trades más cercanos a cualquier evento
    trade_proximities = []
    for trade in trades:
        entry = trade['entry_time']
        min_dist = float('inf')
        closest_group = None
        
        for group, group_events in events_by_group.items():
            for ev in group_events:
                dist = abs((entry - ev['scheduled_at_ny']).total_seconds() / 60)
                if dist < min_dist:
                    min_dist = dist
                    closest_group = group
        
        trade_proximities.append({
            'distance_min': int(min_dist),
            'anchor_group': closest_group,
            'result': trade['result'],
            'pnl_r': trade['pnl_r'],
            'entry_time': trade['entry_time'].isoformat(),
        })
    
    results['closest_trades'] = sorted(trade_proximities, key=lambda x: x['distance_min'])[:30]
    
    return results


def generate_markdown_report(results: dict) -> str:
    """Genera reporte en markdown."""
    lines = []
    lines.append("# NEWS IMPACT ANALYSIS REPORT")
    lines.append(f"\n**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Dataset:** official_anchors (190 eventos) vs trades_realistic.csv")
    lines.append("\n---\n")
    
    # Summary
    lines.append("## RESUMEN EJECUTIVO\n")
    lines.append(f"- **Total eventos anchor:** {results['summary']['total_events']}")
    lines.append(f"- **Total trades analizados:** {results['summary']['total_trades']} (2020-2025)")
    lines.append(f"- **Trades por año:** {dict(results['summary']['trades_by_year'])}")
    lines.append("\n---\n")
    
    # Distribución por anchor_group
    lines.append("## 1. DISTRIBUCIÓN DE EVENTOS POR GRUPO\n")
    for group, stats in sorted(results['by_anchor_group'].items()):
        lines.append(f"### {group}")
        lines.append(f"- Eventos: {stats['total_events']}")
        lines.append(f"- Años: {stats['years']}")
        
        # Horas principales
        hours = sorted(stats['hours_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append(f"- Horarios top: {', '.join(f'{h}:00h ({c})' for h, c in hours)}")
        
        # Meses
        months = sorted(stats['months_distribution'].items())
        lines.append(f"- Meses: {len(months)} diferentes")
        lines.append("")
    
    # Baseline
    baseline = results['baseline']
    lines.append("## 2. BASELINE DE TRADES (2020-2025)\n")
    lines.append(f"- **Win Rate global:** {baseline['win_rate']:.1f}%")
    lines.append(f"- **Avg PnL R:** {baseline['avg_pnl_r']:.2f}R")
    lines.append(f"- **Total trades:** {baseline['total_trades']}")
    lines.append("")
    
    # Análisis por ventana
    lines.append("## 3. ANÁLISIS POR VENTANAS DE TIEMPO\n")
    lines.append("Trades que ocurren dentro de X minutos de un evento anchor:\n")
    
    for window in [5, 15, 30, 60]:
        stats = results['by_window'][window]
        lines.append(f"### Ventana ±{window} minutos")
        lines.append(f"- **Trades afectados:** {stats['trades_count']} ({stats['percentage_of_total']:.1f}% del total)")
        
        if stats['trades_count'] > 0:
            lines.append(f"- **Win Rate:** {stats['win_rate']:.1f}% (Δ {stats['delta_win_rate']:+.1f}% vs baseline)")
            lines.append(f"- **Avg PnL R:** {stats['avg_pnl_r']:.2f}R (Δ {stats['delta_pnl']:+.2f}R vs baseline)")
            
            # Por grupo
            lines.append("- **Por anchor group:**")
            for group, gstats in sorted(stats['by_group'].items()):
                if gstats['count'] > 0:
                    lines.append(f"  - {group}: {gstats['count']} trades, WR={gstats['win_rate']:.1f}%, PnL={gstats['avg_pnl']:.2f}R")
        lines.append("")
    
    # Trades más cercanos
    lines.append("## 4. TRADES MÁS CERCANOS A EVENTOS\n")
    lines.append("(Top 20 trades con menor distancia a cualquier evento anchor)\n")
    lines.append("| Min | Grupo | Result | PnL R | Entry Time |")
    lines.append("|-----|-------|--------|-------|------------|")
    for t in results['closest_trades'][:20]:
        et = t['entry_time'][:16] if len(t['entry_time']) > 16 else t['entry_time']
        lines.append(f"| {t['distance_min']} | {t['anchor_group']} | {t['result']} | {t['pnl_r']:.2f} | {et} |")
    
    return "\n".join(lines)


def main():
    print("Loading data...")
    events = load_anchor_events()
    trades = load_trades()
    
    print(f"Events: {len(events)}")
    print(f"Trades (2020-2025): {len(trades)}")
    
    print("Analyzing...")
    results = analyze_impact(trades, events)
    
    print("Generating report...")
    report = generate_markdown_report(results)
    
    # Guardar reporte
    report_path = Path("c:/Users/alera/Desktop/BOT DE TRADING CURSOR/reports/official_anchors/news_impact_analysis_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Guardar JSON
    json_path = Path("c:/Users/alera/Desktop/BOT DE TRADING CURSOR/reports/official_anchors/news_impact_analysis.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"\nReport: {report_path}")
    print(f"JSON: {json_path}")
    
    # Resumen consola
    print("\n=== RESUMEN ===")
    print(f"Trades 2020-2025: {len(trades)}")
    print(f"Baseline WR: {results['baseline']['win_rate']:.1f}%")
    
    for w in [5, 15, 30, 60]:
        s = results['by_window'][w]
        print(f"±{w}min: {s['trades_count']} trades ({s['percentage_of_total']:.1f}%) - WR {s['win_rate']:.1f}%")


if __name__ == "__main__":
    main()
