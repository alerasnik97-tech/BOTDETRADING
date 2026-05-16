import MetaTrader5 as mt5
import time

class MT5OrderRouter:
    def __init__(self, symbol="EURUSD", magic=123456):
        self.symbol = symbol
        self.magic = magic
        
    def send_order(self, action, volume, price=None, sl=None, tp=None, comment="Demo Trade"):
        """Acciones: 'BUY' o 'SELL'"""
        symbol_info = mt5.symbol_info(self.symbol)
        if not symbol_info:
            print(f"Error: {self.symbol} no encontrado.")
            return None
            
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                print(f"Error: fallo al seleccionar {self.symbol}")
                return None

        order_type = mt5.ORDER_TYPE_BUY if action == 'BUY' else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price if price else (mt5.symbol_info_tick(self.symbol).ask if action == 'BUY' else mt5.symbol_info_tick(self.symbol).bid),
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
            "deviation": 20,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Error en envio de orden: {result.retcode} - {result.comment}")
            return None
            
        print(f"Orden ejecutada con exito: Ticket {result.order}")
        return result

    def close_position(self, ticket, comment="Close position"):
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            print(f"No se encontro la posicion con ticket {ticket}")
            return False
            
        pos = positions[0]
        tick = mt5.symbol_info_tick(self.symbol)
        
        type_dict = {
            mt5.POSITION_TYPE_BUY: mt5.ORDER_TYPE_SELL,
            mt5.POSITION_TYPE_SELL: mt5.ORDER_TYPE_BUY
        }
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "pos": ticket,
            "position": ticket,
            "volume": pos.volume,
            "type": type_dict[pos.type],
            "price": tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask,
            "deviation": 20,
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Error al cerrar posicion: {result.retcode}")
            return False
        return True
