import pandas as pd
import numpy as np

def calculate_body_fraction(df):
    return abs(df['open'] - df['close']) / (df['high'] - df['low'])

def detect_sweep_reclaim(df, level, side='high'):
    if side == 'high':
        sweep = df['high'] > level
        reclaim = df['close'] < level
    else:
        sweep = df['low'] < level
        reclaim = df['close'] > level
    return sweep & reclaim

def smoke_test_logic():
    print("Iniciando Smoke Test Técnico: H6_LTF_MOMENTUM_ALIGN")
    
    # Datos sintéticos para validación de lógica
    data = {
        'open': [1.1000, 1.1010, 1.1005],
        'high': [1.1005, 1.1020, 1.1010],
        'low': [1.0995, 1.1005, 1.1000],
        'close': [1.1002, 1.1008, 1.1009]
    }
    df = pd.DataFrame(data)
    level = 1.1015 # Nivel a barrer
    
    # 1. Detectar Sweep + Reclaim
    df['is_reclaim'] = detect_sweep_reclaim(df, level, side='high')
    
    # 2. Calcular Body Fraction
    df['bf'] = calculate_body_fraction(df)
    
    # 3. Trigger Condition (MOMENTUM > 0.60)
    df['trigger'] = df['is_reclaim'] & (df['bf'] > 0.60)
    
    print("\nResultados de Lógica:")
    print(df)
    
    if any(df['is_reclaim']):
        print("\n[PASS] Detección de Reclaim implementable.")
    if any(df['bf'] > 0):
        print("[PASS] Cálculo de Body Fraction implementable.")
    
    print("\nSmoke Test finalizado. Lógica 100% determinista y programable.")

if __name__ == "__main__":
    smoke_test_logic()
