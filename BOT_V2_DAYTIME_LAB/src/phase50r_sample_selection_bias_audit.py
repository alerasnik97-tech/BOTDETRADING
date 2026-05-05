import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

# --- CONFIGURACIÓN ---
BASE_DIR = r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo"
RAW_TRADES_PATH = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "outputs", "phase38_manipulante_deep_explainer", "csv", "phase38_raw_trades_enriched.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "BOT_V2_DAYTIME_LAB", "reports", "manipulante_tick_historical")

AUDITED_MONTHS = ["2024-05", "2024-06", "2024-07", "2024-08", "2024-10", "2024-11", "2025-01", "2025-03", "2025-07"]

def calculate_metrics(df):
    if df.empty:
        return pd.Series({'trades_count': 0, 'total_R': 0, 'PF': 0, 'Winrate': 0, 'Expectancy': 0})
    
    total_r = df['r_result'].sum()
    wins = df[df['r_result'] > 0]['r_result'].sum()
    losses = abs(df[df['r_result'] < 0]['r_result'].sum())
    pf = wins / losses if losses > 0 else (wins if wins > 0 else 0)
    wr = (len(df[df['r_result'] > 0]) / len(df)) * 100
    exp = total_r / len(df)
    
    return pd.Series({
        'trades_count': len(df),
        'total_R': total_r,
        'PF': pf,
        'Winrate': wr,
        'Expectancy': exp,
        'TP': len(df[df['outcome'] == 'TP']),
        'BE': len(df[df['outcome'] == 'BE']),
        'SL': len(df[df['outcome'] == 'SL']),
        'forced_close': len(df[df['status'] == 'FORCED_CLOSE'])
    })

def main():
    print("Iniciando Auditoría de Sesgo de Selección (Phase 50R)...")
    df = pd.read_csv(RAW_TRADES_PATH)
    
    # Asegurar formato de fecha
    df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True)
    df['year_month'] = df['entry_time'].dt.strftime('%Y-%m')
    
    # 1. Inventario Mensual Completo
    inventory = df.groupby('year_month').apply(calculate_metrics).reset_index()
    
    def classify(row):
        if row['trades_count'] < 5: return "LOW_SAMPLE"
        if row['total_R'] > 5: return "GOOD"
        if row['total_R'] < -3: return "ADVERSO"
        if row['PF'] < 0.5: return "VERY_ADVERSE"
        return "NORMAL"
    
    inventory['classification'] = inventory.apply(classify, axis=1)
    inventory.to_csv(os.path.join(REPORTS_DIR, "PHASE50R_FULL_HISTORICAL_MONTH_INVENTORY.csv"), index=False)
    print(f"Inventario generado: {len(inventory)} meses encontrados.")

    # 2. Comparativa Auditados vs No Auditados
    inventory['is_audited'] = inventory['year_month'].isin(AUDITED_MONTHS)
    bias_report = inventory.groupby('is_audited').agg({
        'PF': 'mean',
        'Expectancy': 'mean',
        'Winrate': 'mean',
        'total_R': 'mean',
        'year_month': 'count'
    }).rename(columns={'year_month': 'month_count'}).reset_index()
    
    bias_report.to_csv(os.path.join(REPORTS_DIR, "PHASE50R_AUDITED_VS_UNAUDITED_BIAS_REPORT.csv"), index=False)
    
    # 3. Meses Adversos Obligatorios
    worst_r = inventory.sort_values('total_R').head(5)['year_month'].tolist()
    worst_pf = inventory[inventory['trades_count'] >= 5].sort_values('PF').head(5)['year_month'].tolist()
    high_sl = inventory.sort_values('SL', ascending=False).head(5)['year_month'].tolist()
    
    mandatory_adverse = list(set(worst_r + worst_pf + high_sl))
    with open(os.path.join(REPORTS_DIR, "PHASE50R_MANDATORY_ADVERSE_MONTHS.json"), 'w') as f:
        json.dump(mandatory_adverse, f, indent=4)
    print(f"Meses adversos obligatorios identificados: {len(mandatory_adverse)}")

    # 4. Monte Carlo
    print("Ejecutando Simulación Monte Carlo (1000 iteraciones)...")
    sim_results = []
    n_months = len(AUDITED_MONTHS)
    all_months = inventory['year_month'].unique()
    
    for _ in range(1000):
        sample = np.random.choice(all_months, n_months, replace=False)
        metrics = calculate_metrics(df[df['year_month'].isin(sample)])
        sim_results.append(metrics)
    
    df_sim = pd.DataFrame(sim_results)
    current_metrics = calculate_metrics(df[df['year_month'].isin(AUDITED_MONTHS)])
    
    # Percentiles
    percentile_pf = (df_sim['PF'] < current_metrics['PF']).mean() * 100
    percentile_r = (df_sim['total_R'] < current_metrics['total_R']).mean() * 100
    
    mc_report = pd.DataFrame([{
        'metric': 'PF',
        'current_value': current_metrics['PF'],
        'sim_mean': df_sim['PF'].mean(),
        'percentile': percentile_pf
    }, {
        'metric': 'Total_R',
        'current_value': current_metrics['total_R'],
        'sim_mean': df_sim['total_R'].mean(),
        'percentile': percentile_r
    }])
    
    mc_report.to_csv(os.path.join(REPORTS_DIR, "PHASE50R_SAMPLE_SELECTION_MONTE_CARLO.csv"), index=False)
    print(f"Monte Carlo completado. Percentil PF: {percentile_pf:.2f}%")

if __name__ == "__main__":
    main()
