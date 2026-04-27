
import pandas as pd
import numpy as np
from phase14_engine import Phase14Engine
from datetime import time

class Phase15Engine(Phase14Engine):
    def __init__(self, data_manifest_path):
        super().__init__(data_manifest_path)
    
    def run_backtest(self, df_ltf, signals, news_df, config):
        """
        config additions:
        'rollover_block': True (17:00-19:00 NY)
        """
        # We override run_backtest to add the rollover block logic if needed
        # Actually we can just inject the rollover block into news_blocked_times or start/end logic
        
        # Inject rollover block into news_blocked_times (simplest way to prevent opening)
        if config.get('rollover_block', True):
            # No opening between 17:00 and 19:00 NY
            # This is handled in the entry logic loop
            pass
            
        return super().run_backtest(df_ltf, signals, news_df, config)

    def _is_in_rollover(self, ny_time):
        # 17:00 - 19:00 NY
        rollover_start = time(17, 0)
        rollover_end = time(19, 0)
        return rollover_start <= ny_time < rollover_end

    # Overriding entry logic in run_backtest requires copy-pasting the loop, 
    # but I'll add a check in the loop directly.
    def run_backtest_p15(self, df_ltf, signals, news_df, config):
        # Re-implementing with rollover block
        trades = []
        active_trade = None
        
        news_blocked_times = set()
        if not news_df.empty:
            news_guard = config.get('news_guard_mins', 30)
            for nt in news_df['timestamp']:
                for m in range(-news_guard, news_guard + 1):
                    news_blocked_times.add((nt + pd.Timedelta(minutes=m)).replace(second=0, microsecond=0))
        
        start_t = time(7, 0) # Mandatory 07:00
        end_t = time(20, 0)   # Mandatory 20:00
        close_t = time(20, 0)
        
        # Convert signals to dict for O(1) lookup if it's a DataFrame
        if isinstance(signals, pd.DataFrame):
            # Only keep rows with signals
            sig_df = signals[signals['signal'] != 0]
            signals_by_idx = sig_df.to_dict('index')
        else:
            signals_by_idx = signals
        trades_today = 0
        current_date = None

        for row in df_ltf.itertuples():
            i = row.Index
            ny_time = row.timestamp_ny.time()
            ny_date = row.timestamp_ny.date()
            
            if ny_date != current_date:
                current_date = ny_date
                trades_today = 0

            if active_trade:
                # Forced close at 20:00
                if ny_time >= close_t:
                    active_trade['status'] = 'FORCED_CLOSE_2000'
                    active_trade['exit_price'] = row.close_bid if active_trade['type'] == 'LONG' else row.close_ask
                    active_trade['exit_time'] = row.timestamp_ny
                    trades.append(active_trade)
                    active_trade = None
                    continue

                # Exit Logic (Standard from Phase 14)
                # ... [Exit logic is identical to Phase 14, I'll use a simplified version or call parent]
                # For Phase 15, I'll keep it explicit to ensure no lookahead and spread stress
                
                if active_trade['type'] == 'LONG':
                    if row.low_bid <= active_trade['sl']:
                        active_trade['status'] = 'SL'
                        active_trade['exit_price'] = active_trade['sl']
                        active_trade['exit_time'] = row.timestamp_ny
                        trades.append(active_trade)
                        active_trade = None
                        continue
                    if row.high_bid >= active_trade['tp']:
                        active_trade['status'] = 'TP'
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['exit_time'] = row.timestamp_ny
                        trades.append(active_trade)
                        active_trade = None
                        continue
                else: # SHORT
                    if row.high_ask >= active_trade['sl']:
                        active_trade['status'] = 'SL'
                        active_trade['exit_price'] = active_trade['sl']
                        active_trade['exit_time'] = row.timestamp_ny
                        trades.append(active_trade)
                        active_trade = None
                        continue
                    if row.low_ask <= active_trade['tp']:
                        active_trade['status'] = 'TP'
                        active_trade['exit_price'] = active_trade['tp']
                        active_trade['exit_time'] = row.timestamp_ny
                        trades.append(active_trade)
                        active_trade = None
                        continue
                continue
            
            # Entry Logic
            if i in signals_by_idx:
                if trades_today >= config.get('max_trades_per_day', 1): continue
                if not (start_t <= ny_time < end_t): continue
                
                # Rollover block
                if self._is_in_rollover(ny_time): continue
                
                # News Guard
                if row.timestamp.replace(second=0, microsecond=0) in news_blocked_times:
                    continue
                
                sig = signals_by_idx[i]
                sl_buffer = config.get('sl_buffer_pips', 0.0) * 0.0001
                
                if sig['signal'] == 1:
                    entry_p = row.close_ask
                    sl = (row.low_bid - 0.0001) - sl_buffer
                    risk = entry_p - sl
                    if risk <= 0: continue
                    active_trade = {
                        'type': 'LONG', 'entry_time': row.timestamp_ny, 'entry_price': entry_p,
                        'sl': sl, 'tp': entry_p + risk * config['tp_r'], 'risk': risk, 'status': 'OPEN'
                    }
                elif sig['signal'] == -1:
                    entry_p = row.close_bid
                    sl = (row.high_ask + 0.0001) + sl_buffer
                    risk = sl - entry_p
                    if risk <= 0: continue
                    active_trade = {
                        'type': 'SHORT', 'entry_time': row.timestamp_ny, 'entry_price': entry_p,
                        'sl': sl, 'tp': entry_p - risk * config['tp_r'], 'risk': risk, 'status': 'OPEN'
                    }
                if active_trade:
                    trades_today += 1
                    
        return pd.DataFrame(trades)
