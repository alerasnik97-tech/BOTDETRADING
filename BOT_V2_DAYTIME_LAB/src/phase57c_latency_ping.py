import MetaTrader5 as mt5
import time
from datetime import datetime

SYMBOL = "EURUSD"
LOTS = 0.01

def run_latency_test():
    if not mt5.initialize():
        print(f"FAILED: MT5 Initialization failed: {mt5.last_error()}")
        return

    # 1. Check Spread
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        print(f"FAILED: Could not get tick for {SYMBOL}")
        mt5.shutdown()
        return

    spread_points = tick.ask - tick.bid
    spread_pips = spread_points * 10000
    print(f"Current Spread: {spread_pips:.2f} pips")

    if spread_pips > 2.0:
        print(f"ABORTED: Spread too high ({spread_pips:.2f} pips)")
        mt5.shutdown()
        return

    # 2. Execution Ping (BUY)
    print(f"Sending MARKET BUY {LOTS} {SYMBOL}...")
    t0 = time.perf_counter()
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOTS,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "magic": 575757,
        "comment": "PHASE57C_LATENCY_PING",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    t1 = time.perf_counter()
    
    latency_ms = (t1 - t0) * 1000

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"FAILED: Order Send Error: {result.retcode} - {result.comment}")
        mt5.shutdown()
        return

    ticket = result.order
    entry_price = result.price
    print(f"ORDER_FILL_SUCCESS: Ticket {ticket}, Price {entry_price}")
    print(f"LATENCY_RTT: {latency_ms:.2f} ms")

    # 3. Immediate Close (SELL)
    print("Closing position immediately...")
    
    # Refresh tick for close price
    tick_close = mt5.symbol_info_tick(SYMBOL)
    
    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": LOTS,
        "type": mt5.ORDER_TYPE_SELL,
        "position": ticket,
        "price": tick_close.bid,
        "magic": 575757,
        "comment": "PHASE57C_LATENCY_CLOSE",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    close_result = mt5.order_send(close_request)
    if close_result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"WARNING: Close failed! Code {close_result.retcode}. Manual intervention needed.")
    else:
        print(f"CLOSE_SUCCESS: Price {close_result.price}")

    mt5.shutdown()

    # Final Summary for Auditor
    report = {
        "verdict": "PHASE57C_LATENCY_TEST_COMPLETED",
        "latency_ms": round(latency_ms, 2),
        "spread_pips": round(spread_pips, 2),
        "entry_price": entry_price,
        "exit_price": close_result.price if close_result.retcode == mt5.TRADE_RETCODE_DONE else None
    }
    print(f"FINAL_REPORT_JSON: {report}")

if __name__ == "__main__":
    run_latency_test()
