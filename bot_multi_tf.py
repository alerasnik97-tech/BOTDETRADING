import backtrader as bt
import datetime

class MTF_Robust_Strategy(bt.Strategy):
    # Parámetros optimizables
    params = (
        ('macro_ema_fast', 50),
        ('macro_ema_slow', 200),
        ('micro_rsi_period', 14),
        ('rsi_lower', 30),
        ('rsi_upper', 70),
        ('atr_period', 14),
        ('atr_multiplier', 2.0),
        ('risk_per_trade', 0.02) # Arriesgamos el 2% del capital por operación
    )

    def __init__(self):
        # Asignación de datos: data0 es el Micro (ej. 1H), data1 es el Macro (ej. Diario)
        self.micro_data = self.datas[0]
        self.macro_data = self.datas[1]

        # 1. Indicadores Macro (Filtro de Tendencia en Data1)
        self.ema_fast = bt.indicators.EMA(self.macro_data.close, period=self.p.macro_ema_fast)
        self.ema_slow = bt.indicators.EMA(self.macro_data.close, period=self.p.macro_ema_slow)
        self.macro_trend_up = self.ema_fast > self.ema_slow

        # 2. Indicadores Micro (Gatillos de Entrada en Data0)
        self.rsi = bt.indicators.RSI(self.micro_data.close, period=self.p.micro_rsi_period)
        
        # 3. Volatilidad para Gestión de Riesgo (ATR en Data0)
        self.atr = bt.indicators.ATR(self.micro_data, period=self.p.atr_period)

    def next(self):
        # Evitar operar si no tenemos suficientes datos en el timeframe mayor
        if len(self.macro_data) < self.p.macro_ema_slow:
            return

        if not self.position:
            # LÓGICA DE ENTRADA (LONG)
            # Solo operamos si la tendencia macro es alcista
            if self.macro_trend_up[0]:
                # Gatillo: Reversión a la media (oversold) en el timeframe menor
                if self.rsi[0] < self.p.rsi_lower:
                    
                    # Position Sizing Dinámico basado en Volatilidad (ATR)
                    stop_distance = self.atr[0] * self.p.atr_multiplier
                    risk_amount = self.broker.get_value() * self.p.risk_per_trade
                    
                    # Prevenir división por cero en activos sin movimiento
                    if stop_distance > 0:
                        size = risk_amount / stop_distance
                        
                        # Calculamos los niveles de las órdenes Bracket
                        price = self.micro_data.close[0]
                        stop_price = price - stop_distance
                        take_profit = price + (stop_distance * 1.5) # Risk/Reward 1:1.5
                        
                        self.buy_bracket(
                            size=size,
                            price=price,
                            stopprice=stop_price,
                            limitprice=take_profit
                        )
                        print(f"[{self.micro_data.datetime.datetime(0)}] COMPRA EJECUTADA. Precio: {price:.2f} | Size: {size:.2f}")

            # *Aquí el programador experto añadiría la lógica simétrica para cortos (Shorts)*

# --- SETUP DEL MOTOR (CÓMO ALIMENTAR EL BOT) ---
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(MTF_Robust_Strategy)

    # Nota: Como experto, deberás conectar tus datos de YFinance, Binance o tu broker aquí.
    # Es crucial usar feeds de la misma fuente pero con la compresión correcta.
    # Ejemplo conceptual:
    # data_1h = bt.feeds.GenericCSVData(dataname='tu_data_1h_2020_2025.csv', ...)
    # data_daily = bt.feeds.GenericCSVData(dataname='tu_data_daily_2020_2025.csv', ...)
    # cerebro.adddata(data_1h)   # Index 0
    # cerebro.adddata(data_daily)# Index 1

    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001) # 0.1% de comisión típica
    
    print('Capital Inicial: %.2f' % cerebro.broker.getvalue())
    # cerebro.run()
    # print('Capital Final: %.2f' % cerebro.broker.getvalue())
    # cerebro.plot()