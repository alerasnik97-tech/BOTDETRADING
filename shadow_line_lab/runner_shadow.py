import pandas as pd
import numpy as np
from datetime import timedelta
import os

class ShadowRunner:
    def __init__(self, config):
        self.config = config

    def run_daily_check(self, date_str, h1_data, m5_data, news_data, levels):
        """
        Simula la lógica para un día específico.
        """
        results = {
            "date": date_str,
            "line_name": self.config["variant_id"],
            "signal_found": False,
            "news_blocked": False,
            "classification": "NO_SIGNAL",
            "entry": 0.0,
            "sl": 0.0,
            "tp": 0.0,
            "exit_reason": "N/A",
            "pnl_r": 0.0,
            "timeout_flag": False,
            "notes": ""
        }

        # 1. Buscar Sweeps en H1
        for level_name, level_price in levels.items():
            if level_name not in self.config["levels"]:
                continue
            
            # Filtro simplificado de sweeps en H1 para el día
            sweeps = self.detect_h1_sweeps(h1_data, level_price)
            for sweep in sweeps:
                sweep_time = sweep['time']
                
                # 2. Filtro de Noticias
                if self.is_news_blocked(sweep_time, news_data):
                    results["news_blocked"] = True
                    results["notes"] += f"News blocked at {sweep_time} on {level_name}; "
                    continue
                
                # 3. Buscar Confirmación M5 (+0h a +1h)
                confirmation = self.find_m5_confirmation(sweep, m5_data, level_price)
                if confirmation:
                    results["signal_found"] = True
                    results["classification"] = "TRADE_EXECUTED"
                    
                    # 4. Ejecutar Trade
                    trade_result = self.execute_trade(confirmation, sweep, m5_data, level_price)
                    results.update(trade_result)
                    return results # Solo 1 trade por día
                    
        return results

    def detect_h1_sweeps(self, h1_data, level_price):
        sweeps = []
        # Lógica: low < nivel y close > nivel (Long) | high > nivel y close < nivel (Short)
        # Asumimos h1_data es un DataFrame con columnas: time, open, high, low, close
        for idx, row in h1_data.iterrows():
            # Long Sweep
            if row['low'] < level_price and row['close'] > level_price:
                sweeps.append({'time': row['time'], 'type': 'LONG', 'extreme': row['low']})
            # Short Sweep
            elif row['high'] > level_price and row['close'] < level_price:
                sweeps.append({'time': row['time'], 'type': 'SHORT', 'extreme': row['high']})
        return sweeps

    def is_news_blocked(self, sweep_time, news_data):
        if news_data is None or len(news_data) == 0:
            return False
        # Buffer de +/- 30m
        limit_min = sweep_time - timedelta(minutes=self.config["news_filter_minutes"])
        limit_max = sweep_time + timedelta(minutes=self.config["news_filter_minutes"])
        
        # Filtro en news_data (asumimos tiene columna 'datetime')
        blocking_news = news_data[(news_data['datetime'] >= limit_min) & (news_data['datetime'] <= limit_max)]
        return len(blocking_news) > 0

    def find_m5_confirmation(self, sweep, m5_data, level_price):
        # Ventana +0h a +1h post sweep
        win_start = sweep['time']
        win_end = sweep['time'] + timedelta(hours=1)
        
        window_m5 = m5_data[(m5_data['time'] >= win_start) & (m5_data['time'] < win_end)]
        
        for idx, row in window_m5.iterrows():
            if sweep['type'] == 'LONG':
                # Reclaim: close > nivel (después de haber estado debajo)
                if row['close'] > level_price:
                    return {'time': row['time'], 'type': 'LONG'}
            else:
                # Reclaim: close < nivel
                if row['close'] < level_price:
                    return {'time': row['time'], 'type': 'SHORT'}
        return None

    def execute_trade(self, confirmation, sweep, m5_data, level_price):
        # Entrada: apertura vela siguiente
        # Buscamos la vela siguiente a la confirmación
        next_candles = m5_data[m5_data['time'] > confirmation['time']].sort_values('time')
        if next_candles.empty:
            return {"classification": "ABORTED_NO_NEXT_CANDLE"}
        
        entry_candle = next_candles.iloc[0]
        entry_price = entry_candle['open']
        
        if confirmation['type'] == 'LONG':
            entry_price += (self.config["long_entry_buffer"] / 10000.0)
            sl = sweep['extreme'] - (self.config["sl_buffer"] / 10000.0)
            risk = entry_price - sl
            if (risk * 10000.0) < self.config["min_risk_pips"]:
                return {"classification": "ABORTED_MIN_RISK"}
            tp = entry_price + (risk * self.config["tp_r"])
        else:
            entry_price -= (self.config["short_entry_buffer"] / 10000.0)
            sl = sweep['extreme'] + (self.config["sl_buffer"] / 10000.0)
            risk = sl - entry_price
            if (risk * 10000.0) < self.config["min_risk_pips"]:
                return {"classification": "ABORTED_MIN_RISK"}
            tp = entry_price - (risk * self.config["tp_r"])

        # Monitorear salida (TP, SL, Timeout)
        timeout_limit = entry_candle['time'] + timedelta(hours=self.config["timeout_hours"])
        monitoring_data = m5_data[m5_data['time'] >= entry_candle['time']].sort_values('time')
        
        for idx, row in monitoring_data.iterrows():
            # Check Timeout
            if row['time'] >= timeout_limit:
                # Salida al precio de cierre de la vela de timeout
                pnl = (row['close'] - entry_price) / risk if confirmation['type'] == 'LONG' else (entry_price - row['close']) / risk
                return {
                    "entry": entry_price, "sl": sl, "tp": tp, 
                    "exit_reason": "TIMEOUT", "pnl_r": round(pnl, 4), "timeout_flag": True
                }
            
            # Check SL/TP
            if confirmation['type'] == 'LONG':
                if row['low'] <= sl:
                    return {"entry": entry_price, "sl": sl, "tp": tp, "exit_reason": "SL", "pnl_r": -1.0}
                if row['high'] >= tp:
                    return {"entry": entry_price, "sl": sl, "tp": tp, "exit_reason": "TP", "pnl_r": self.config["tp_r"]}
            else:
                if row['high'] >= sl:
                    return {"entry": entry_price, "sl": sl, "tp": tp, "exit_reason": "SL", "pnl_r": -1.0}
                if row['low'] <= tp:
                    return {"entry": entry_price, "sl": sl, "tp": tp, "exit_reason": "TP", "pnl_r": self.config["tp_r"]}
        
        return {"classification": "OPEN_OR_UNKNOWN_EXIT"}
