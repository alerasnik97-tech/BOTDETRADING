"""Ledger <-> dossier reconciliation gate.

Pure, dependency-light verification used to FAIL fast when engine/report metrics
do not reconcile with the trade ledger. Runs without any backtest, strategy run,
optimization, sweep, validation, holdout, news or 2025/2026 data: it only inspects
already-produced artifacts or synthetic fixtures.

A "violation" is a dict: {code, detail}. An empty list means the ledger reconciles.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

REL_TOL = 1e-6
ABS_TOL = 1e-6


def _is_close(a: float, b: float, rel: float = REL_TOL, abs_: float = ABS_TOL) -> bool:
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def reconcile_trades(trades: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Per-trade sign / result invariants.

    Each trade needs: direction, entry_price, exit_price, pnl_usd, pnl_r,
    exit_reason. result is optional (recomputed from pnl_usd if absent).
    """
    violations: list[dict[str, str]] = []
    for idx, t in enumerate(trades):
        pnl_usd = float(t["pnl_usd"])
        pnl_r = float(t["pnl_r"])
        direction = str(t["direction"]).strip().lower()
        exit_reason = str(t.get("exit_reason", "")).strip().lower()

        # pnl_r and pnl_usd must share sign (pnl_r = pnl_usd / risk_usd, risk_usd > 0)
        if pnl_usd != 0.0 and pnl_r != 0.0 and (pnl_usd > 0) != (pnl_r > 0):
            violations.append({"code": "PNL_SIGN_MISMATCH",
                               "detail": f"trade#{idx}: pnl_usd={pnl_usd} pnl_r={pnl_r} opposite signs"})

        # stop_loss must not be a profit; take_profit must not be a loss.
        if exit_reason == "stop_loss" and pnl_usd > ABS_TOL:
            violations.append({"code": "STOP_LOSS_POSITIVE_PNL",
                               "detail": f"trade#{idx}: exit_reason=stop_loss but pnl_usd={pnl_usd}>0"})
        if exit_reason == "take_profit" and pnl_usd < -ABS_TOL:
            violations.append({"code": "TAKE_PROFIT_NEGATIVE_PNL",
                               "detail": f"trade#{idx}: exit_reason=take_profit but pnl_usd={pnl_usd}<0"})

        # result label (if present) must derive from pnl sign.
        if "result" in t and t["result"] is not None:
            res = str(t["result"]).strip().lower()
            expected = "win" if pnl_usd > 0 else ("loss" if pnl_usd < 0 else "breakeven")
            if res != expected:
                violations.append({"code": "RESULT_LABEL_MISMATCH",
                                   "detail": f"trade#{idx}: result={res} but pnl_usd={pnl_usd} => {expected}"})
    return violations


def reconcile_equity(
    trades: Sequence[dict[str, Any]],
    equity_series: Sequence[float],
    starting_equity: float,
    reported_total_return_pct: float | None = None,
    reported_max_drawdown_pct: float | None = None,
) -> list[dict[str, str]]:
    """Equity / drawdown / total-return must derive from the ledger."""
    violations: list[dict[str, str]] = []
    sum_pnl = sum(float(t["pnl_usd"]) - float(t.get("entry_commission_usd", 0.0)) for t in trades)
    additive_end = starting_equity + sum_pnl

    if equity_series:
        end_eq = float(equity_series[-1])
        # Ending equity can only diverge from additive if a documented compounding
        # model applies; an unexplained > 1% gap is a hard violation.
        if not _is_close(end_eq, additive_end, rel=1e-3, abs_=1e-2):
            violations.append({"code": "ENDING_EQUITY_DECOUPLED",
                               "detail": f"equity_end={end_eq:.4f} vs start+Σpnl={additive_end:.4f}"})

        peak = equity_series[0]
        max_dd = 0.0
        for e in equity_series:
            e = float(e)
            peak = max(peak, e)
            if peak > 0:
                max_dd = max(max_dd, (peak - e) / peak * 100.0)
        # A net-losing ledger must show a real drawdown; an all-zero dd is dead.
        if sum_pnl < -ABS_TOL and max_dd <= ABS_TOL:
            violations.append({"code": "DRAWDOWN_DEAD",
                               "detail": f"Σpnl={sum_pnl:.2f}<0 but max_dd_from_equity={max_dd:.6f}~0"})
        if reported_max_drawdown_pct is not None and not _is_close(
            reported_max_drawdown_pct, max_dd, rel=1e-2, abs_=1e-2
        ):
            violations.append({"code": "MAX_DRAWDOWN_MISMATCH",
                               "detail": f"reported={reported_max_drawdown_pct} recomputed={max_dd:.4f}"})

    if reported_total_return_pct is not None and starting_equity:
        recomputed_ret = (sum_pnl / starting_equity) * 100.0
        # Sign must agree: a negative ledger cannot yield a positive total return.
        if (recomputed_ret > 0) != (reported_total_return_pct > 0) and abs(
            reported_total_return_pct - recomputed_ret
        ) > 1.0:
            violations.append({"code": "TOTAL_RETURN_SIGN_DECOUPLED",
                               "detail": f"reported={reported_total_return_pct} ledger_additive={recomputed_ret:.4f}"})
    return violations


def reconcile_summary(
    profit_factor: float,
    expectancy_r: float,
    total_return_pct: float,
) -> list[dict[str, str]]:
    """summary.json internal coherence: PF<1 & expectancy<0 cannot coexist with a
    positive total return absent an explicit, verifiable compounding explanation."""
    violations: list[dict[str, str]] = []
    if profit_factor < 1.0 and expectancy_r < 0.0 and total_return_pct > 0.0:
        violations.append({"code": "SUMMARY_SELF_CONTRADICTION",
                           "detail": f"PF={profit_factor}<1 & expectancy={expectancy_r}<0 but total_return={total_return_pct}>0"})
    return violations


def reconcile_cost_profiles(profiles: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    """Each profile must self-report its own name; profiles must not be duplicates
    unless their config genuinely matches."""
    violations: list[dict[str, str]] = []
    for name, meta in profiles.items():
        reported = str(meta.get("cost_profile", "")).strip().lower()
        if reported and reported != name.strip().lower():
            violations.append({"code": "COST_PROFILE_MISLABEL",
                               "detail": f"folder '{name}' self-reports cost_profile='{reported}'"})
    names = list(profiles)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = profiles[names[i]], profiles[names[j]]
            same_cfg = a.get("cost_profile") == b.get("cost_profile") and a.get(
                "execution_mode"
            ) == b.get("execution_mode")
            if same_cfg:
                violations.append({"code": "COST_PROFILE_DUPLICATE",
                                   "detail": f"'{names[i]}' and '{names[j]}' share identical cost config"})
    return violations


def reconcile_all(**kwargs: Any) -> list[dict[str, str]]:
    """Convenience aggregator. Pass any subset of the section kwargs."""
    v: list[dict[str, str]] = []
    if "trades" in kwargs and "equity_series" not in kwargs:
        v += reconcile_trades(kwargs["trades"])
    if "equity_series" in kwargs:
        v += reconcile_trades(kwargs["trades"])
        v += reconcile_equity(
            kwargs["trades"],
            kwargs["equity_series"],
            kwargs["starting_equity"],
            kwargs.get("reported_total_return_pct"),
            kwargs.get("reported_max_drawdown_pct"),
        )
    if {"profit_factor", "expectancy_r", "total_return_pct"} <= kwargs.keys():
        v += reconcile_summary(
            kwargs["profit_factor"], kwargs["expectancy_r"], kwargs["total_return_pct"]
        )
    if "profiles" in kwargs:
        v += reconcile_cost_profiles(kwargs["profiles"])
    return v
