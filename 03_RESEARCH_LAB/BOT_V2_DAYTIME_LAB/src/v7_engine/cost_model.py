from __future__ import annotations
from dataclasses import dataclass, asdict

@dataclass
class CostModelConfig:
    commission_per_lot_round_turn: float = 0.0
    commission_per_trade_r: float = 0.10
    slippage_pips: float = 0.0
    min_commission: float = 0.0
    mode: str = "conservative"  # "conservative" | "lot_based" | "ftmo" | "zero"
    
    # Configuraciones requeridas para modo FTMO
    instrument: str = "EURUSD"
    account_currency: str = "USD"
    lot_size_units: float = 100000.0
    pip_size: float = 0.0001
    pip_value_per_standard_lot_usd: float = 10.0
    risk_per_trade_pct: float = 1.0

class UnknownCostModeError(Exception):
    """Excepción levantada programáticamente cuando se especifica un régimen de costos desconocido."""
    pass

class CostCalculationBlockedError(Exception):
    """Excepción cuando faltan datos críticos (como SL) impidiendo calcular costos."""
    pass

class CostModel:
    """
    Modelo centralizado de costos institucionales y fricciones de ejecución.
    Aplica deducciones explícitas de comisión y deslizamiento sobre el PnL y R transaccionales.
    """
    def __init__(self, config: CostModelConfig | None = None):
        self.config = config if config is not None else CostModelConfig()
        if self.config.mode not in ["conservative", "lot_based", "ftmo", "zero"]:
            raise UnknownCostModeError(f"Régimen de costos '{self.config.mode}' no soportado.")

    def serialize_config(self) -> dict[str, any]:
        """Serializa los parámetros inmutables del modelo para su persistencia en reportes OOS."""
        return asdict(self.config)

    def apply_costs_to_trade(
        self,
        gross_r: float,
        reason: str = "TP",
        lot_size: float = 0.0,
        risk_per_trade_cash: float = 1000.0,
        sl_pips: float | None = None
    ) -> dict[str, float]:
        """
        Deduce programáticamente las fricciones de la orden finalizada, retornando
        el desglose exhaustivo de gross_r, commission_r, slippage_r y net_r.
        """
        commission_r = 0.0
        commission_usd = 0.0
        slippage_r = 0.0

        if self.config.mode == "ftmo":
            if self.config.instrument != "EURUSD":
                raise CostCalculationBlockedError(f"Instrumento {self.config.instrument} no soportado sin config explícita en modo FTMO.")
            if sl_pips is None or sl_pips <= 0:
                raise CostCalculationBlockedError("Falta SL (sl_pips) para calcular la comisión FTMO basada en lotes dinámicos.")
                
            # Fórmula estricta FTMO para EURUSD
            # commission_r = commission_usd_per_lot / (sl_pips * pip_value_per_standard_lot_usd)
            commission_r = self.config.commission_per_lot_round_turn / (sl_pips * self.config.pip_value_per_standard_lot_usd)
            
            # Cálculo de USD aproximado para auditoría, aunque no cambie commission_r
            # risk_usd = account_equity_usd * risk_per_trade_pct / 100 (asumimos risk_per_trade_cash es el risk_usd)
            lots = risk_per_trade_cash / (sl_pips * self.config.pip_value_per_standard_lot_usd)
            commission_usd = lots * self.config.commission_per_lot_round_turn
            
        elif self.config.mode == "conservative":
            commission_r = self.config.commission_per_trade_r
        elif self.config.mode == "lot_based":
            if lot_size > 0 and risk_per_trade_cash > 0:
                cash_comm = lot_size * self.config.commission_per_lot_round_turn
                cash_comm = max(cash_comm, self.config.min_commission)
                commission_r = cash_comm / risk_per_trade_cash
                commission_usd = cash_comm
            else:
                commission_r = self.config.commission_per_trade_r
        elif self.config.mode == "zero":
            commission_r = 0.0

        # Determinación del deslizamiento (slippage) en unidades de riesgo (R)
        if self.config.slippage_pips > 0:
            if sl_pips is not None and sl_pips > 0:
                slippage_r = self.config.slippage_pips / sl_pips
            else:
                # Fallback si no hay sl_pips explícito
                slippage_r = self.config.slippage_pips * 0.10

        # La comisión y slippage aplican sobre TODA salida (TP, SL, BE, TIME)
        net_r = gross_r - commission_r - slippage_r
        
        return {
            "gross_r": round(gross_r, 4),
            "sl_pips": round(sl_pips, 2) if sl_pips is not None else 0.0,
            "lots": round(risk_per_trade_cash / (sl_pips * self.config.pip_value_per_standard_lot_usd), 2) if sl_pips and self.config.mode == "ftmo" else 0.0,
            "commission_usd": round(commission_usd, 4),
            "commission_r": round(commission_r, 4),
            "slippage_r": round(slippage_r, 4),
            "net_r": round(net_r, 4)
        }
