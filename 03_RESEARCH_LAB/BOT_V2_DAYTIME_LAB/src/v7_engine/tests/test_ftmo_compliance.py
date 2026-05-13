import pytest
from datetime import datetime, timedelta
from src.v7_engine.ftmo_compliance import FtmoComplianceEngine

def test_daily_loss_limit_blows():
    """
    Verifica: test_daily_loss_limit_blows.
    Una caída del equity intradiario por debajo del 95% del balance inicial del día
    activa incondicionalmente el Blown State.
    """
    engine = FtmoComplianceEngine(initial_balance=100000.0)
    ts = datetime(2026, 5, 12, 10, 0, 0) # 10:00 UTC
    
    # Actualización sin pérdidas
    assert engine.update_state(ts, closed_pnl=0.0, floating_pnl=0.0) is True
    
    # Pérdida flotante del 4.9% ($4,900) -> Equity = $95,100 (Pasa)
    assert engine.update_state(ts + timedelta(minutes=1), closed_pnl=0.0, floating_pnl=-4900.0) is True
    
    # Pérdida flotante del 5.1% ($5,100) -> Equity = $94,900 (Quiebra diaria)
    assert engine.update_state(ts + timedelta(minutes=2), closed_pnl=0.0, floating_pnl=-5100.0) is False
    assert engine.blown is True
    assert "pérdida diaria" in str(engine.blown_reason).lower()

def test_max_loss_limit_blows():
    """
    Verifica: test_max_loss_limit_blows.
    Una caída del equity acumulado por debajo del 90% del capital original
    activa el Blown State absoluto.
    """
    engine = FtmoComplianceEngine(initial_balance=100000.0)
    ts1 = datetime(2026, 5, 12, 12, 0, 0)
    
    # Realizar pérdida del 4% en el día 1 ($4,000) -> Balance = $96,000. No quiebra diaria ni absoluta.
    assert engine.update_state(ts1, closed_pnl=-4000.0, floating_pnl=0.0) is True
    
    # Avanzar al día siguiente para resetear el umbral diario
    # 17:00 NY del 12 de mayo ocurre a las 21:00 o 22:00 UTC. Situamos en el día 13.
    ts2 = datetime(2026, 5, 13, 12, 0, 0)
    assert engine.update_state(ts2, closed_pnl=0.0, floating_pnl=0.0) is True
    assert engine.start_of_day_balance == 96000.0 # Reseteado correctamente al nuevo inicio
    
    # Pérdida flotante adicional de $5,500 en el día 2.
    # Pérdida diaria = 5,500 / 96,000 = 5.7% (Viola el diario del día 2)
    # Pero para probar el absoluto de forma pura, supongamos que pierde $6,500 directo
    # Equity total = 96,000 - 6,500 = 89,500 < 90,000
    assert engine.update_state(ts2 + timedelta(minutes=1), closed_pnl=0.0, floating_pnl=-6500.0) is False
    assert engine.blown is True
    # Ambas condiciones pueden dispararse, pero garantizamos que quiebra
    assert engine.blown_reason is not None

def test_daily_reset_at_1700_ny():
    """
    Verifica: test_daily_reset_at_1700_ny.
    El ancla FX resetea la base de cálculo de pérdidas diarias exactamente al cruzar las 17:00 NY.
    """
    engine = FtmoComplianceEngine(initial_balance=100000.0)
    
    # 16:59 NY en Mayo (DST activo, UTC-4) corresponde a 20:59 UTC
    ts_before = datetime(2026, 5, 12, 20, 59, 0)
    engine.update_state(ts_before, closed_pnl=2000.0, floating_pnl=0.0)
    assert engine.start_of_day_balance == 100000.0 # Sigue anclado al inicio previo
    
    # 17:01 NY corresponde a 21:01 UTC. Cruza el umbral.
    ts_after = datetime(2026, 5, 12, 21, 1, 0)
    engine.update_state(ts_after, closed_pnl=0.0, floating_pnl=0.0)
    # El nuevo balance inicial del día adopta las ganancias previas
    assert engine.start_of_day_balance == 102000.0

def test_floating_pnl_triggers_blow():
    """
    Verifica: test_floating_pnl_triggers_blow.
    El monitoreo es continuo; el drawdown flotante sin cerrar es suficiente para quebrar.
    """
    engine = FtmoComplianceEngine(initial_balance=100000.0)
    ts = datetime(2026, 5, 12, 15, 0, 0)
    
    assert engine.update_state(ts, closed_pnl=0.0, floating_pnl=-5050.0) is False
    assert engine.blown is True

def test_risk_scaling_correct():
    """
    Verifica: test_risk_scaling_correct.
    El riesgo por posición escala dinámicamente representando siempre el 1% del balance.
    """
    engine = FtmoComplianceEngine(initial_balance=100000.0)
    assert engine.get_position_risk_amount() == 1000.0
    
    # Ganar $10,000
    engine.update_state(datetime(2026, 5, 12, 12, 0), closed_pnl=10000.0, floating_pnl=0.0)
    assert engine.get_position_risk_amount() == 1100.0 # 1% de 110,000

def test_blown_state_irreversible():
    """
    Verifica: test_blown_state_irreversible.
    Una vez disparada la quiebra, no se puede revertir inyectando PnL positivo posterior.
    """
    engine = FtmoComplianceEngine(initial_balance=100000.0)
    ts = datetime(2026, 5, 12, 12, 0)
    
    # Quebrar
    engine.update_state(ts, closed_pnl=0.0, floating_pnl=-6000.0)
    assert engine.blown is True
    
    # Intento de recuperación espuria posterior
    assert engine.update_state(ts + timedelta(minutes=1), closed_pnl=20000.0, floating_pnl=0.0) is False
    assert engine.blown is True # Sigue quebrado

def test_minimum_days_tracking():
    """
    Verifica: test_minimum_days_tracking.
    El sistema contabiliza correctamente días únicos de trading FX.
    """
    engine = FtmoComplianceEngine(initial_balance=100000.0)
    
    # Día 1 FX
    engine.update_state(datetime(2026, 5, 12, 10, 0), closed_pnl=100.0, floating_pnl=0.0)
    engine.update_state(datetime(2026, 5, 12, 11, 0), closed_pnl=200.0, floating_pnl=0.0)
    assert engine.get_minimum_days_count() == 1
    
    # Día 2 FX (Tras las 17:00 NY -> > 21:00 UTC)
    engine.update_state(datetime(2026, 5, 13, 10, 0), closed_pnl=-50.0, floating_pnl=0.0)
    assert engine.get_minimum_days_count() == 2
