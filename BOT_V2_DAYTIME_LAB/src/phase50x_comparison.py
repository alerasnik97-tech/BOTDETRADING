import pandas as pd
import json

recalc = pd.read_csv(r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50X_OPERATIONAL_1945_MONTHLY_METRICS.csv')
p50w = pd.read_csv(r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50W_CLOSE_TIME_ALIGNMENT_METRICS.csv')

results = []
months = ['2017-05', '2017-08', '2020-04', '2024-10']

for m in months:
    row = {'month': m}
    # Policy 19:45 (Locked)
    r_row = recalc[recalc['month'] == m]
    if not r_row.empty:
        row['Policy_1945_R'] = r_row['total_R'].values[0]
        row['Policy_1945_PF'] = r_row['PF'].values[0]
    
    # Phase50W 19:45 (Previous estimate)
    w_row = p50w[(p50w['month'] == m) & (p50w['model'] == '19:45')]
    if not w_row.empty:
        row['W_1945_R'] = w_row['total_R'].values[0]
    
    # Original Phase50S (From my memory or files)
    # 2017-05: 8.82, 2017-08: 5.47, 2020-04: 12.49, 2024-10: 11.24
    orig = {'2017-05': 8.82, '2017-08': 5.47, '2020-04': 12.49, '2024-10': 11.24}
    row['Original_S_R'] = orig.get(m, 0)
    
    results.append(row)

final = pd.DataFrame(results)
final.to_csv(r'C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\BOT_V2_DAYTIME_LAB\reports\manipulante_tick_historical\PHASE50X_POLICY_LOCK_COMPARISON.csv', index=False)
print(final)
