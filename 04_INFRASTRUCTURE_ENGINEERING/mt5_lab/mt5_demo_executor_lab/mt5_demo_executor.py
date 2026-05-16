import time
import MetaTrader5 as mt5
from datetime import datetime, timezone
import pandas as pd

from mt5_data_bridge import MT5DataBridge
from mt5_order_router import MT5OrderRouter
from mt5_timeout_manager import MT5TimeoutManager
from mt5_news_guard import MT5NewsGuard
from mt5_risk_engine import MT5RiskEngine
from mt5_kill_switch import MT5KillSwitch
from mt5_demo_telemetry import MT5DemoTelemetry
from mt5_auto_shutdown import MT5AutoShutdown

class MT5DemoExecutor:
    def __init__(self):
        self.bridge = MT5DataBridge()
        self.router = MT5OrderRouter()
        self.timeout_mgr = MT5TimeoutManager()
        self.news_guard = MT5NewsGuard(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\data\news_eurusd_am_fortress_v3.csv")
        self.risk_engine = MT5RiskEngine()
        self.kill_switch = MT5KillSwitch()
        self.telemetry = MT5DemoTelemetry(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\mt5_demo_executor_lab\outputs")
        self.shutdown_mgr = MT5AutoShutdown(r"C:\Users\alera\Desktop\Bot\BOT DE TRADING ultimo\mt5_demo_executor_lab\config\mt5_demo_runtime_config.json", self.timeout_mgr)
        
        self.trades_today = 0
        self.initial_balance = 0
        self.is_running = True
        self.last_heartbeat = None

    def _get_mt5_server_time(self):
        """Intenta obtener la hora del servidor desde el terminal MT5"""
        if not mt5.symbol_select(self.router.symbol, True):
            return None
        tick = mt5.symbol_info_tick(self.router.symbol)
        if tick:
            return datetime.fromtimestamp(tick.time, tz=timezone.utc)
        return None

    def _perform_heartbeat(self):
        """Registra el estado actual en logs, consola y JSON"""
        now_utc = datetime.now(timezone.utc)
        now_ny = datetime.now(self.shutdown_mgr.tz_ny)
        server_time = self._get_mt5_server_time()
        
        account_info = mt5.account_info()
        tick = mt5.symbol_info_tick(self.router.symbol)
        positions = mt5.positions_get(symbol=self.router.symbol)
        
        instruction = self.shutdown_mgr.get_shutdown_instruction()
        within_window = (instruction == "CONTINUE_RUNNING")
        
        # 1. Log CSV
        heartbeat_details = (
            f"NY:{now_ny.strftime('%H:%M:%S')} | "
            f"Server:{server_time.strftime('%H:%M:%S') if server_time else 'N/A'} | "
            f"Spread:{((tick.ask - tick.bid)/0.0001 if tick else 0.0):.1f} | "
            f"Pos:{len(positions) if positions else 0} | "
            f"Window:{within_window}"
        )
        self.telemetry.log_event("HEARTBEAT", heartbeat_details)
        
        # 2. Console
        print(f"[HEARTBEAT] {datetime.now().strftime('%H:%M:%S')} | "
              f"Demo OK | NY={now_ny.strftime('%H:%M')} | "
              f"Server={server_time.strftime('%H:%M') if server_time else 'N/A'} | "
              f"Positions={len(positions) if positions else 0} | "
              f"Next shutdown={self.shutdown_mgr.config['stop_time_ny']} NY")
        
        # 3. Status JSON
        status = {
            "executor_status": "RUNNING",
            "account_type": "DEMO" if account_info and account_info.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else "UNKNOWN",
            "account_login": account_info.login if account_info else 0,
            "broker": account_info.company if account_info else "N/A",
            "symbol": self.router.symbol,
            "ny_time_now": now_ny.isoformat(),
            "mt5_server_time_now": server_time.isoformat() if server_time else None,
            "within_runtime_window": within_window,
            "auto_shutdown_time_ny": self.shutdown_mgr.config['stop_time_ny'],
            "open_positions_count": len(positions) if positions else 0,
            "last_heartbeat_at": now_utc.isoformat()
        }
        self.telemetry.save_status(status)
        self.last_heartbeat = now_utc

    def _compute_levels(self, h1_df):
        """Calcula niveles Asia, London y PDH/PDL de forma simplificada para el loop live"""
        # Nota: En un entorno live, esto se recalcula cada hora o sesion.
        # Por brevedad, aqui simulamos la extraccion del ultimo PDH/PDL y sesiones.
        levels = {}
        # PDH/PDL del dia anterior
        yesterday = h1_df.iloc[-48:-24] # Aproximacion
        levels['pdh'] = yesterday['high'].max()
        levels['pdl'] = yesterday['low'].min()
        
        # Asia (18:00 - 02:00 NY)
        # Esto requiere filtrar por horas en NY. El bridge ya nos da time_ny como indice.
        asia_bars = h1_df[h1_df.index.hour.isin([18, 19, 20, 21, 22, 23, 0, 1])]
        if not asia_bars.empty:
             levels['asia_h'] = asia_bars['high'].max()
             levels['asia_l'] = asia_bars['low'].min()
             
        return levels

    def run(self):
        if not self.bridge.connect():
            return
            
        self.initial_balance = mt5.account_info().balance
        self.telemetry.log_event("SESSION_START", f"Balance inicial: {self.initial_balance}")
        
        try:
            while self.is_running:
                now_utc = datetime.now(timezone.utc)
                
                # 0. Heartbeat (Cada 5 minutos o primer ciclo)
                if self.last_heartbeat is None or (now_utc - self.last_heartbeat).total_seconds() >= 300:
                    self._perform_heartbeat()

                # 1. Kill Switch Checks
                if self.kill_switch.check_conditions(self.initial_balance):
                    self.telemetry.log_event("KILL_SWITCH_TRIGGERED", "Drawdown o error critico")
                    break
                    
                if self.kill_switch.check_spread("EURUSD"):
                    time.sleep(10)
                    continue

                # 2. Timeout Management
                self.timeout_mgr.check_timeouts(self.router)
                
                # 2.1 Auto Shutdown Check
                instruction = self.shutdown_mgr.get_shutdown_instruction()
                if instruction != "CONTINUE_RUNNING":
                    if instruction == "SAFE_TO_SHUTDOWN":
                        self.telemetry.log_event("END_OF_DAY_SAFE_SHUTDOWN")
                        self.is_running = False
                        break
                    elif instruction == "SHUTDOWN_AFTER_FORCED_DEMO_CLOSE":
                        self.telemetry.log_event("DEMO_TIMEOUT_OR_EOD_CLOSE", "Cierre forzado de fin de dia")
                        self.timeout_mgr.check_timeouts(self.router) # Forzamos cierre
                        self.is_running = False
                        break
                    elif instruction == "SHUTDOWN_DELAYED_POSITION_OPEN":
                        # Seguimos corriendo pero avisamos
                        self.telemetry.log_event("SHUTDOWN_DELAYED_POSITION_OPEN", "Esperando cierre de posicion < 4h")
                
                # 3. Decision Logic (Cada 1 minuto o nueva vela M5)
                # Obtenemos datos
                h1 = self.bridge.get_latest_rates(mt5.TIMEFRAME_H1, 100)
                m5 = self.bridge.get_latest_rates(mt5.TIMEFRAME_M5, 20)
                
                if h1.empty or m5.empty:
                    time.sleep(30)
                    continue
                    
                levels = self._compute_levels(h1)
                
                # 4. Escanear Sweeps (Simplificado para el demo lab)
                # En un entorno real, aqui llamariamos a la logica del truth_model
                last_h1 = h1.iloc[-1]
                for level_name, level_val in levels.items():
                    # Deteccion de sweep basica en la vela cerrada anterior
                    prev_h1 = h1.iloc[-2]
                    is_sweep_high = prev_h1['high'] > level_val and prev_h1['close'] < level_val
                    is_sweep_low = prev_h1['low'] < level_val and prev_h1['close'] > level_val
                    
                    if is_sweep_high or is_sweep_low:
                        # 5. Risk & News Checks
                        if not self.risk_engine.is_risk_ok(self.trades_today):
                            continue
                        if self.news_guard.is_blocked(now_utc):
                            continue
                            
                        # 6. Reclaim M5 (Vela actual M5)
                        last_m5 = m5.iloc[-1]
                        action = None
                        if is_sweep_high and last_m5['close'] < level_val: # Reclaim short
                            action = 'SELL'
                            sl = prev_h1['high'] + 0.0001 # +1 pip
                            risk_pips = (sl - last_m5['close']) / 0.0001
                            tp = last_m5['close'] - (risk_pips * 1.5 * 0.0001)
                        elif is_sweep_low and last_m5['close'] > level_val: # Reclaim long
                            action = 'BUY'
                            sl = prev_h1['low'] - 0.0001 # -1 pip
                            risk_pips = (last_m5['close'] - sl) / 0.0001
                            tp = last_m5['close'] + (risk_pips * 1.5 * 0.0001)
                            
                        if action:
                            lot = self.risk_engine.calculate_lot(self.initial_balance, risk_pips)
                            res = self.router.send_order(action, lot, sl=sl, tp=tp, comment=f"Sweep {level_name}")
                            if res:
                                self.trades_today += 1
                                self.telemetry.log_trade(res.order, action, res.price, sl, tp)
                                
                # Guardar status
                self.telemetry.save_status({
                    "last_tick_utc": now_utc.isoformat(),
                    "trades_today": self.trades_today,
                    "levels": levels
                })
                
                time.sleep(60) # Esperar 1 minuto para el siguiente ciclo
                
        except KeyboardInterrupt:
            print("Detencion manual solicitada.")
        finally:
            self.bridge.disconnect()
            self.telemetry.log_event("DEMO_EXECUTOR_STOPPED", "Conexion MT5 cerrada")
            # Guardar status final
            self.telemetry.save_status({
                "session_date": datetime.now().strftime("%Y-%m-%d"),
                "stopped_at_ny": datetime.now(self.shutdown_mgr.tz_ny).isoformat(),
                "final_status": "DEMO_SESSION_CLOSED_CLEAN",
                "trades_today": self.trades_today
            })

if __name__ == "__main__":
    executor = MT5DemoExecutor()
    executor.run()
