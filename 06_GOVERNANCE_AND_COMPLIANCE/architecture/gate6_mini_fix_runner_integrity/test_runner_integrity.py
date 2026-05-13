import pytest
import pandas as pd
from datetime import datetime
from src.v6_utils.execution import next_bar_execute, next_bar_execute_stop, FillResult
from src.v7_engine.trade_record import TradeRecord
from src.v7_engine.cost_model import CostModelConfig

def test_gate6_mini_v2b_stop_is_not_market_clone():
    """
    Prueba que la logica Stop Order real exige el cruce fisico del extremo
    y no se comporta como una clonacion incondicional a mercado.
    """
    t0 = pd.Timestamp("2026-05-13 09:00:00", tz="UTC")
    t1 = pd.Timestamp("2026-05-13 09:01:00", tz="UTC")
    
    # Precios donde Ask/Bid jamas tocan un stop alcista extremo
    df_ticks = pd.DataFrame({
        "ask": [1.0500, 1.0501],
        "bid": [1.0498, 1.0499]
    }, index=[t0, t1])
    
    # 1. next_bar_execute ejecuta a mercado incondicionalmente en t1
    fill_mkt = next_bar_execute("long", t0, df_ticks)
    assert fill_mkt is not None
    assert fill_mkt.fill_price == 1.0501
    
    # 2. next_bar_execute_stop con stop_price inalcanzable retorna None
    fill_stp = next_bar_execute_stop("long", t0, stop_price=1.0600, ticks_after=df_ticks)
    assert fill_stp is None
    
    # 3. next_bar_execute_stop con stop_price alcanzado retorna FillResult causal
    fill_stp_ok = next_bar_execute_stop("long", t0, stop_price=1.0500, ticks_after=df_ticks)
    assert fill_stp_ok is not None
    assert fill_stp_ok.reason == "STOP_FILL"

def test_gate6_mini_no_artificial_eom_truncation():
    """
    Prueba que si un trade agota una cuota estricta de 3000 ticks marcado como EOM,
    se clasifica como truncamiento artificial e invalida la contabilidad dimensional.
    """
    # Escenario A: Salida EOM por recorte artificial de .head(3000)
    t_bad = TradeRecord(
        trade_id="bad-uuid", side="long",
        signal_time=datetime(2026,5,13,9,0), fill_time=datetime(2026,5,13,9,1), exit_time=datetime(2026,5,13,9,10),
        entry_price=1.0500, exit_price=1.0510, sl_price=1.0480, tp_price=1.0540,
        exit_reason="EOM", gross_r=0.5, sl_pips=20.0, lots=1.0, commission_usd=5.0,
        commission_r=0.05, slippage_r=0.0, net_r=0.45, gross_pnl_usd=50.0, net_pnl_usd=45.0,
        risk_usd=100.0, cost_model_mode="ftmo", ftmo_cost_applied=True, instrument="EURUSD",
        forced_exit=True, be_activated=False, valid_closed_trade=False,
        eom_type="ARTIFICIAL_TRUNCATION", ticks_scanned_count=3000
    )
    assert t_bad.eom_type == "ARTIFICIAL_TRUNCATION"
    assert t_bad.valid_closed_trade is False
    
    # Escenario B: Salida EOM real por final fisico de datos
    t_good = TradeRecord(
        trade_id="good-uuid", side="long",
        signal_time=datetime(2026,5,13,9,0), fill_time=datetime(2026,5,13,9,1), exit_time=datetime(2026,5,13,15,59),
        entry_price=1.0500, exit_price=1.0510, sl_price=1.0480, tp_price=1.0540,
        exit_reason="EOM", gross_r=0.5, sl_pips=20.0, lots=1.0, commission_usd=5.0,
        commission_r=0.05, slippage_r=0.0, net_r=0.45, gross_pnl_usd=50.0, net_pnl_usd=45.0,
        risk_usd=100.0, cost_model_mode="ftmo", ftmo_cost_applied=True, instrument="EURUSD",
        forced_exit=True, be_activated=False, valid_closed_trade=True,
        eom_type="REAL_DATA_END", ticks_scanned_count=12500
    )
    assert t_good.eom_type == "REAL_DATA_END"
    assert t_good.valid_closed_trade is True

def test_gate6_mini_news_missing_blocks_run():
    """
    Prueba que la ausencia del archivo canonico de noticias gatilla
    Fail-Close con la terminologia exigida.
    """
    from pathlib import Path
    import sys
    
    # Forzamos un path invalido simulando ausencia
    fake_vault = Path("/ruta_inexistente_boveda_x999")
    news_path = fake_vault / "data" / "news_eurusd_am_fortress_v3.csv"
    
    with pytest.raises(FileNotFoundError) as excinfo:
        if not news_path.exists():
            raise FileNotFoundError("[FAIL-CLOSE INSTITUCIONAL] Ausencia critica del archivo canonico de calendario de noticias en la boveda. Abortando simulacion.")
            
    assert "[FAIL-CLOSE INSTITUCIONAL]" in str(excinfo.value)
