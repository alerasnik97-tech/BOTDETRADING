import MetaTrader5 as mt5

class MT5RiskEngine:
    def __init__(self, risk_per_trade=0.001, max_trades_per_day=1):
        self.risk_per_trade = risk_per_trade
        self.max_trades_per_day = max_trades_per_day
        
    def calculate_lot(self, balance, risk_pips, symbol="EURUSD"):
        """Calcula el lote basado en el riesgo (0.10%) y el stop en pips"""
        # Simplificacion para EURUSD en cuenta USD: 1 pip en 1 lote = $10
        # Formula: Lote = (Balance * Riesgo) / (Riesgo_Pips * Valor_Pip_Lote)
        risk_amount = balance * self.risk_per_trade
        if risk_pips <= 0:
            return 0.01
            
        lot = risk_amount / (risk_pips * 10)
        # Normalizar al minimo permitido
        lot = max(0.01, round(lot, 2))
        return lot

    def is_risk_ok(self, current_day_trades):
        if current_day_trades >= self.max_trades_per_day:
            print("RIESGO: Limite de trades diarios alcanzado.")
            return False
            
        positions = mt5.positions_get()
        if positions and len(positions) >= 1:
            print("RIESGO: Ya existe una posicion abierta.")
            return False
            
        return True
