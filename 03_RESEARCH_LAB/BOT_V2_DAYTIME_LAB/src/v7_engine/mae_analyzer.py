from __future__ import annotations
import pandas as pd
import numpy as np

class MaeAnalyzer:
    """
    Analizador programático ejecutable de Excursión Adversa (MAE) y Favorable (MFE)
    para el cierre de reservas de selección adversa (GAP-002 Platinum).
    Calcula distribuciones granulares tick a tick e impone veto automático ante estados patológicos.
    """
    def __init__(
        self,
        pathological_threshold_gt_09r: float = 0.25,
        watch_threshold_gt_09r: float = 0.10
    ):
        # Umbrales inmutables prefijados antes de evaluar cualquier estrategia candidata
        self.patho_thresh = pathological_threshold_gt_09r
        self.watch_thresh = watch_threshold_gt_09r

    def analyze_trade_ticks(
        self,
        trade_id: str | int,
        side: str,
        entry_time: pd.Timestamp,
        exit_time: pd.Timestamp,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        result_net_r: float,
        ticks_df: pd.DataFrame,
        entry_type: str = "market"
    ) -> dict[str, any]:
        """
        Analiza el recorrido causal físico de una orden consumiendo el flujo contiguo de ticks
        y erradicando el atajo optimista de mechas de velas agregadas.
        """
        # Filtrar ticks estrictamente comprendidos durante la vida útil de la posición
        path = ticks_df[(ticks_df.index >= entry_time) & (ticks_df.index <= exit_time)]
        
        r_size = abs(entry_price - sl_price)
        if r_size < 1e-7:
            r_size = 0.0001

        max_adverse_p = entry_price
        max_favorable_p = entry_price
        
        if not path.empty:
            if side == "long":
                # Excursión adversa en compras se evalúa contra el mínimo Bid alcanzado
                max_adverse_p = float(path["bid"].min())
                # Excursión favorable se evalúa contra el máximo Bid alcanzado
                max_favorable_p = float(path["bid"].max())
            else:
                # Excursión adversa en ventas se evalúa contra el máximo Ask alcanzado
                max_adverse_p = float(path["ask"].max())
                # Excursión favorable se evalúa contra el mínimo Ask alcanzado
                max_favorable_p = float(path["ask"].min())

        # Cálculos puros en R
        if side == "long":
            mae_r = (entry_price - max_adverse_p) / r_size
            mfe_r = (max_favorable_p - entry_price) / r_size
        else:
            mae_r = (max_adverse_p - entry_price) / r_size
            mfe_r = (entry_price - max_favorable_p) / r_size

        mae_r = max(0.0, round(mae_r, 4))
        mfe_r = max(0.0, round(mfe_r, 4))
        
        is_winner = result_net_r > 0
        is_loser = result_net_r < 0
        
        # Categorización por cubetas de estrés
        if mae_r >= 0.90:
            bucket = "extreme_stress"
        elif mae_r >= 0.75:
            bucket = "high_stress"
        elif mae_r >= 0.50:
            bucket = "moderate_stress"
        else:
            bucket = "low_stress"

        patho_flag = is_winner and (mae_r >= 0.90)

        return {
            "trade_id": trade_id,
            "side": side,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "sl_price": sl_price,
            "tp_price": tp_price,
            "r_size": r_size,
            "mae_r": mae_r,
            "mfe_r": mfe_r,
            "result_net_r": result_net_r,
            "is_winner": is_winner,
            "is_loser": is_loser,
            "entry_type": entry_type,
            "max_adverse_price": max_adverse_p,
            "max_favorable_price": max_favorable_p,
            "mae_bucket": bucket,
            "pathological_flag": patho_flag
        }

    def generate_summary(self, analyzed_trades: list[dict[str, any]]) -> dict[str, any]:
        """
        Agrega las distribuciones individuales, computa percentiles y emite el estatus
        incondicional de la estrategia con capacidad de veto sobre PASS_STRONG.
        """
        if not analyzed_trades:
            return {
                "mean_mae_winners": 0.0, "mean_mae_losers": 0.0, "median_mae": 0.0,
                "p75_mae": 0.0, "p90_mae": 0.0, "pct_mae_gt_0_9r": 0.0, "pct_mae_gt_0_75r": 0.0,
                "adverse_selection_score": 0.0, "correlation_mae_result": 0.0,
                "status": "HEALTHY", "pass_strong_vetoed": False
            }

        maes = np.array([t["mae_r"] for t in analyzed_trades])
        results = np.array([t["result_net_r"] for t in analyzed_trades])
        
        winners_mae = [t["mae_r"] for t in analyzed_trades if t["is_winner"]]
        losers_mae = [t["mae_r"] for t in analyzed_trades if t["is_loser"]]

        mean_w = float(np.mean(winners_mae)) if winners_mae else 0.0
        mean_l = float(np.mean(losers_mae)) if losers_mae else 0.0
        
        median_mae = float(np.median(maes))
        p75 = float(np.percentile(maes, 75))
        p90 = float(np.percentile(maes, 90))

        # Porcentajes críticos evaluados prioritariamente sobre el subconjunto ganador
        # para detectar si el sistema depende de ser llenado al borde del colapso
        n_winners = len(winners_mae)
        gt_09 = sum(1 for m in winners_mae if m >= 0.90)
        gt_075 = sum(1 for m in winners_mae if m >= 0.75)
        
        pct_09 = round(gt_09 / n_winners, 4) if n_winners > 0 else 0.0
        pct_075 = round(gt_075 / n_winners, 4) if n_winners > 0 else 0.0

        # Score de selección adversa
        adv_score = round(mean_w * pct_09 * 10.0, 4)
        
        # Correlación lineal entre excursión adversa y retorno final
        corr = 0.0
        if len(maes) > 1 and np.std(maes) > 1e-5 and np.std(results) > 1e-5:
            corr_mat = np.corrcoef(maes, results)
            corr = round(float(corr_mat[0, 1]), 4)
            if np.isnan(corr):
                corr = 0.0

        # Dictamen inmutable de régimen
        status = "HEALTHY"
        if pct_09 > self.patho_thresh or (mean_w >= 0.85 and n_winners > 5):
            status = "PATHOLOGICAL"
        elif pct_09 > self.watch_thresh or pct_075 > 0.40:
            status = "WATCH"

        # Veto automático sobre la señal PASS_STRONG
        vetoed = status == "PATHOLOGICAL"

        return {
            "mean_mae_winners": round(mean_w, 4),
            "mean_mae_losers": round(mean_l, 4),
            "median_mae": round(median_mae, 4),
            "p75_mae": round(p75, 4),
            "p90_mae": round(p90, 4),
            "pct_mae_gt_0_9r": pct_09,
            "pct_mae_gt_0_75r": pct_075,
            "adverse_selection_score": adv_score,
            "correlation_mae_result": corr,
            "status": status,
            "pass_strong_vetoed": vetoed
        }
