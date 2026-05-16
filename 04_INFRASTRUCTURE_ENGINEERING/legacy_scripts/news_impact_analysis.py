"""
NEWS IMPACT ANALYSIS
Cruza dataset official_anchors (190 eventos) con trades del laboratorio.
Mide impacto en ventanas 5/15/30/60 minutos.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


def parse_datetime(dt_str: str) -> datetime:
    """Parsea datetime en formato '2022-01-24 05:00:00'."""
    return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")


def load_anchor_events() -> list[dict]:
    """Carga eventos desde canonical_anchor_events.csv."""
    csv_path = Path("c:/Users/alera/Desktop/BOT DE TRADING CURSOR/data/official_anchors/out/canonical_anchor_events.csv")
    events = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parsear scheduled_at_ny: "2024-01-11T08:30:00-05:00"
            ny_time_str = row.get('scheduled_at_ny', '')
            if ny_time_str:
                try:
                    # Quitar timezone
                    base_time = ny_time_str[:19]
                    dt = datetime.strptime(base_time, "%Y-%m-%dT%H:%M:%S")
                    events.append({
                        'event_id': row['event_id'],
                        'title': row['title'],
                        'anchor_group': row['anchor_group'],
                        'scheduled_at_ny': dt,
                        'currency': row['currency'],
                        'importance': row['importance'],
                        'source_url': row.get('source_url', ''),
                    })
                except Exception:
                    continue
    return events


def load_trades() -> list[dict]:
    """Carga trades desde trades_realistic.csv."""
    csv_path = Path("c:/Users/alera/Desktop/BOT DE TRADING CURSOR/trades_realistic.csv")
    trades = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                entry_time = parse_datetime(row['entry_time_ny'])
                exit_time = parse_datetime(row['exit_time_ny'])
                trades.append({
                    'pair': row['pair'],
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'direction': row['direction'],
                    'pnl_r': float(row['pnl_r']),
                    'pnl_usd': float(row['pnl_usd']),
                    'result': row['result'],
                    'exit_reason': row['exit_reason'],
                    'sl': float(row['sl']),
                    'tp': float(row['tp']),
                    'entry_price': float(row['entry_price']),
                    'exit_price': float(row['exit_price']),
                })
            except Exception:
                continue
    return trades


def analyze_time_distance(trade_entry: datetime, event_time: datetime) -> int:
    """Calcula distancia en minutos entre trade entry y evento."""
    diff = trade_entry - event_time
    return int(abs(diff.total_seconds()) / 60)


def analyze_impact(trades: list[dict], events: list[dict]) -> dict:
    """Analiza impacto de eventos sobre trades en ventanas de tiempo."""
    
    # Agrupar eventos por anchor_group
    events_by_group = defaultdict(list)
    for ev in events:
        events_by_group[ev['anchor_group']].append(ev)
    
    # Ventanas a analizar (en minutos)
    windows = [5, 15, 30, 60]
    
    results = {
        'global_metrics': {},
        'by_anchor_group': {},
        'by_window': {},
        'trade_event_proximity': [],
    }
    
    # Métricas globales por anchor_group
    for group, group_events in events_by_group.items():
        results['by_anchor_group'][group] = {
            'total_events': len(group_events),
            'years': sorted(set(ev['scheduled_at_ny'].year for ev in group_events)),
            'months_distribution': defaultdict(int),
            'hours_distribution': defaultdict(int),
        }
        for ev in group_events:
            month_key = ev['scheduled_at_ny'].strftime("%Y-%m")
            hour_key = ev['scheduled_at_ny'].hour
            results['by_anchor_group'][group]['months_distribution'][month_key] += 1
            results['by_anchor_group'][group]['hours_distribution'][hour_key] += 1
    
    # Análisis por ventana de tiempo
    for window in windows:
        window_stats = {
            'total_trades_in_window': 0,
            'trades_by_group': defaultdict(list),
            'win_rate': 0,
            'avg_pnl_r': 0,
            'avg_pnl_usd': 0,
        }
        
        trades_in_window = []
        for trade in trades:
            trade_entry = trade['entry_time']
            closest_event = None
            min_distance = float('inf')
            closest_group = None
            
            for group, group_events in events_by_group.items():
                for ev in group_events:
                    dist = analyze_time_distance(trade_entry, ev['scheduled_at_ny'])
                    if dist < min_distance:
                        min_distance = dist
                        closest_event = ev
                        closest_group = group
            
            if min_distance <= window:
                trade_with_context = {**trade, 
                    'distance_to_event': min_distance,
                    'closest_anchor_group': closest_group,
                    'event_time': closest_event['scheduled_at_ny'] if closest_event else None,
                }
                trades_in_window.append(trade_with_context)
                window_stats['trades_by_group'][closest_group].append(trade_with_context)
        
        window_stats['total_trades_in_window'] = len(trades_in_window)
        
        if trades_in_window:
            wins = sum(1 for t in trades_in_window if t['result'] == 'win')
            window_stats['win_rate'] = wins / len(trades_in_window) * 100
            window_stats['avg_pnl_r'] = sum(t['pnl_r'] for t in trades_in_window) / len(trades_in_window)
            window_stats['avg_pnl_usd'] = sum(t['pnl_usd'] for t in trades_in_window) / len(trades_in_window)
            
            # Stats por grupo
            for group, group_trades in window_stats['trades_by_group'].items():
                group_wins = sum(1 for t in group_trades if t['result'] == 'win')
                window_stats[f'{group}_win_rate'] = group_wins / len(group_trades) * 100 if group_trades else 0
                window_stats[f'{group}_count'] = len(group_trades)
        
        results['by_window'][window] = window_stats
    
    # Análisis de proximidad (todos los trades)
    all_proximities = []
    for trade in trades:
        trade_entry = trade['entry_time']
        min_distance = float('inf')
        closest_group = None
        
        for group, group_events in events_by_group.items():
            for ev in group_events:
                dist = analyze_time_distance(trade_entry, ev['scheduled_at_ny'])
                if dist < min_distance:
                    min_distance = dist
                    closest_group = group
        
        all_proximities.append({
            'trade_id': f"{trade['pair']}_{trade_entry}",
            'entry_time': trade_entry.isoformat(),
            'result': trade['result'],
            'pnl_r': trade['pnl_r'],
            'closest_anchor_group': closest_group,
            'distance_minutes': min_distance,
        })
    
    results['trade_event_proximity'] = sorted(all_proximities, key=lambda x: x['distance_minutes'])
    
    return results


def generate_report(results: dict, trades: list, events: list) -> str:
    """Genera reporte en formato markdown."""
    
    lines = []
    lines.append("# NEWS IMPACT ANALYSIS REPORT")
    lines.append(f"\n**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Trades analizados:** {len(trades)}")
    lines.append(f"**Eventos anchor:** {len(events)}")
    lines.append("\n---")
    
    # Métricas por anchor_group
    lines.append("\n## 1. DISTRIBUCIÓN POR ANCHOR GROUP\n")
    for group, stats in sorted(results['by_anchor_group'].items()):
        lines.append(f"### {group}")
        lines.append(f"- Total eventos: {stats['total_events']}")
        lines.append(f"- Años cubiertos: {', '.join(map(str, stats['years']))}")
        
        # Top horas
        hours = sorted(stats['hours_distribution'].items(), key=lambda x: x[1], reverse=True)[:3]
        lines.append(f"- Horarios principales: {', '.join(f'{h}:00 ({c})' for h, c in hours)}")
        lines.append("")
    
    # Análisis por ventana
    lines.append("\n## 2. ANÁLISIS POR VENTANAS DE TIEMPO\n")
    lines.append("Trades que caen dentro de X minutos de un evento anchor:\n")
    
    baseline_win_rate = sum(1 for t in trades if t['result'] == 'win') / len(trades) * 100
    baseline_avg_pnl = sum(t['pnl_r'] for t in trades) / len(trades)
    
    lines.append(f"**Baseline (todos los trades):** Win Rate={baseline_win_rate:.1f}%, Avg PnL R={baseline_avg_pnl:.2f}\n")
    
    for window in [5, 15, 30, 60]:
        stats = results['by_window'][window]
        lines.append(f"### Ventana ±{window} minutos")
        lines.append(f"- Trades en ventana: {stats['total_trades_in_window']} / {len(trades)} ({stats['total_trades_in_window']/len(trades)*100:.1f}%)")
        if stats['total_trades_in_window'] > 0:
            lines.append(f"- Win Rate en ventana: {stats['win_rate']:.1f}%")
            lines.append(f"- Avg PnL R en ventana: {stats['avg_pnl_r']:.2f}")
            lines.append(f"- Avg PnL USD en ventana: ${stats['avg_pnl_usd']:.2f}")
            
            # Comparación vs baseline
            win_rate_diff = stats['win_rate'] - baseline_win_rate
            pnl_diff = stats['avg_pnl_r'] - baseline_avg_pnl
            lines.append(f"- Delta vs baseline: WR {win_rate_diff:+.1f}%, PnL {pnl_diff:+.2f}R")
            
            # Distribución por grupo
            for group in ['FOMC', 'ECB', 'CPI', 'PPI', 'NFP']:
                count = stats.get(f'{group}_count', 0)
                if count > 0:
                    wr = stats.get(f'{group}_win_rate', 0)
                    lines.append(f"  - {group}: {count} trades, WR={wr:.1f}%")
        lines.append("")
    
    # Trades más cercanos a eventos
    lines.append("\n## 3. TRADES MÁS CERCANOS A EVENTOS\n")
    closest_20 = results['trade_event_proximity'][:20]
    lines.append("| Dist (min) | Grupo | Result | PnL R | Entry Time |")
    lines.append("|------------|-------|--------|-------|------------|")
    for t in closest_20:
        lines.append(f"| {t['distance_minutes']} | {t['closest_anchor_group']} | {t['result']} | {t['pnl_r']:.2f} | {t['entry_time'][:16]} |")
    
    return "\n".join(lines)


def main():
    print("Loading anchor events...")
    events = load_anchor_events()
    print(f"Loaded {len(events)} anchor events")
    
    print("Loading trades...")
    trades = load_trades()
    print(f"Loaded {len(trades)} trades")
    
    print("Analyzing impact...")
    results = analyze_impact(trades, events)
    
    print("Generating report...")
    report = generate_report(results, trades, events)
    
    # Guardar reporte
    report_path = Path("c:/Users/alera/Desktop/BOT DE TRADING CURSOR/reports/official_anchors/news_impact_analysis_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Report saved: {report_path}")
    
    # Guardar JSON de resultados
    json_path = Path("c:/Users/alera/Desktop/BOT DE TRADING CURSOR/reports/official_anchors/news_impact_analysis.json")
    
    # Convertir datetime a string para JSON
    json_results = {
        'by_anchor_group': {
            k: {
                **{key: val for key, val in v.items() if key not in ['months_distribution', 'hours_distribution']},
                'months_distribution': dict(v['months_distribution']),
                'hours_distribution': dict(v['hours_distribution']),
            }
            for k, v in results['by_anchor_group'].items()
        },
        'by_window': {
            str(k): {
                **{key: val for key, val in v.items() if not isinstance(val, defaultdict)},
                'trades_by_group': {gk: len(gv) for gk, gv in v['trades_by_group'].items()}
            }
            for k, v in results['by_window'].items()
        },
        'closest_20_trades': results['trade_event_proximity'][:20],
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"JSON results saved: {json_path}")
    
    print("\n=== RESUMEN EJECUTIVO ===")
    print(f"Eventos anchor: {len(events)}")
    print(f"Trades analizados: {len(trades)}")
    
    for group, stats in sorted(results['by_anchor_group'].items()):
        print(f"  {group}: {stats['total_events']} eventos")
    
    print("\nTrades en ventanas:")
    for window in [5, 15, 30, 60]:
        count = results['by_window'][window]['total_trades_in_window']
        pct = count / len(trades) * 100
        print(f"  ±{window}min: {count} trades ({pct:.1f}%)")


if __name__ == "__main__":
    main()
