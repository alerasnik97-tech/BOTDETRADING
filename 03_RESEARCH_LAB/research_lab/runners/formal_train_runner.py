"""Formal train-only runner — versioned, committable, fail-closed.

Replaces the git-ignored `scratch/formal_run_tp01.py` as the ONLY sanctioned,
sealable mechanism to regenerate formal train-only dossiers. Pure infrastructure:
it orchestrates the already-fixed engine / cost-routing / reconciliation layers
and never duplicates strategy or engine logic.

Hard guarantees:
  * Importing this module has NO side effects (no run, no data, no heavy import).
  * `run_single_strategy_formal_train_only` is DRY-RUN by default
    (`request.execute is False`) and aborts before any backtest.
  * Fail-closed: holdout / sealed_holdout / 2025-26 / validation / optimization /
    sweep / news / high-precision / `precision` profile / non-train-only are rejected.
  * Output must live under `.../BOT_V2_DAYTIME_LAB/reports/formal_train_only/...`;
    project root / data vault / production / incubation / scratch / ZIP rejected.
  * Heavy artifacts route under `local_outputs_do_not_commit/<profile>`.
  * No run may be sealed unless the reconciliation gate returns zero violations.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
from datetime import date
from pathlib import PurePosixPath
from typing import Any

from research_lab.config import (
    EngineConfig,
    resolved_cost_profile,
    with_execution_mode,
)
from research_lab import metric_reconciliation as mr

# ----------------------------------------------------------------------------
# Institutional constants
# ----------------------------------------------------------------------------
REPORTS_AREA_PARTS = ("03_RESEARCH_LAB", "BOT_V2_DAYTIME_LAB", "reports", "formal_train_only")
HEAVY_SUBDIR = "local_outputs_do_not_commit"
DATA_VAULT_PART = "05_MARKET_DATA_VAULT"
REQUIRED_TRAIN_DATA_TOKEN = "prepared_train_2015_2024"
FIRST_FORBIDDEN_DATE = date(2025, 1, 1)  # train-only ends 2024-12-31
BANNED_EXECUTION_MODES = ("high_precision_mode",)
BANNED_COST_PROFILES = ("precision", "auto")  # not valid for a sealed formal train run
FORBIDDEN_OUTPUT_PARTS = ("05_MARKET_DATA_VAULT", "production", "incubation", "scratch")

# (folder name, execution_mode, cost_profile) — the only sanctioned plan.
PROFILE_PLAN: tuple[tuple[str, str, str], ...] = (
    ("base", "normal_mode", "base"),
    ("conservative", "conservative_mode", "conservative"),
    ("stress", "stress_mode", "stress"),
)


class RunnerSafetyError(RuntimeError):
    """Raised when a request violates a hard safety / output / scope contract."""


class ReconciliationGateError(RuntimeError):
    """Raised when a run cannot be sealed because the gate found violations."""


@dataclass(frozen=True)
class FormalRunRequest:
    strategy_name: str
    start_date: str
    end_date: str
    data_path: str
    output_dir: str
    train_only: bool = True
    execute: bool = False
    holdout: bool = False
    validation: bool = False
    optimization: bool = False
    sweep: bool = False
    high_precision: bool = False
    news: bool = False


# ----------------------------------------------------------------------------
# Cost-profile wiring
# ----------------------------------------------------------------------------
def build_cost_profile_configs(base_config: EngineConfig) -> dict[str, EngineConfig]:
    """Exactly the 3 sanctioned profiles, each correctly routed (no precision)."""
    configs: dict[str, EngineConfig] = {}
    for name, mode, cost_profile in PROFILE_PLAN:
        cfg = dataclasses.replace(base_config, execution_mode=mode, cost_profile=cost_profile)
        configs[name] = with_execution_mode(cfg, mode)
    return configs


def validate_cost_profile_configs(configs: dict[str, EngineConfig]) -> None:
    expected = {name: (mode, cp) for name, mode, cp in PROFILE_PLAN}
    if set(configs) != set(expected):
        raise RunnerSafetyError(f"expected exactly {sorted(expected)}, got {sorted(configs)}")

    seen: set[tuple[str, str]] = set()
    for name, cfg in configs.items():
        exp_mode, exp_cp = expected[name]
        rp = resolved_cost_profile(cfg)
        if cfg.execution_mode != exp_mode:
            raise RunnerSafetyError(f"{name}: execution_mode={cfg.execution_mode} != {exp_mode}")
        if rp != exp_cp:
            raise RunnerSafetyError(f"{name}: resolved cost_profile={rp} != {exp_cp} (mislabel)")
        if cfg.execution_mode in BANNED_EXECUTION_MODES:
            raise RunnerSafetyError(f"{name}: high-precision mode forbidden in formal train")
        if rp in BANNED_COST_PROFILES:
            raise RunnerSafetyError(f"{name}: cost_profile {rp!r} forbidden in formal train")
        key = (cfg.execution_mode, rp)
        if key in seen:
            raise RunnerSafetyError(f"duplicate cost profile config: {key}")
        seen.add(key)

    cons, strs = configs["conservative"], configs["stress"]
    if not (1.0 < cons.conservative_spread_multiplier < strs.stress_spread_multiplier):
        raise RunnerSafetyError("spread multipliers must satisfy base < conservative < stress")
    if not (1.0 < cons.conservative_slippage_multiplier < strs.stress_slippage_multiplier):
        raise RunnerSafetyError("slippage multipliers must satisfy base < conservative < stress")


# ----------------------------------------------------------------------------
# Scope / output policy
# ----------------------------------------------------------------------------
def _parse_iso(d: str) -> date:
    try:
        return date.fromisoformat(str(d).strip())
    except ValueError as exc:
        raise RunnerSafetyError(f"invalid ISO date: {d!r}") from exc


def validate_train_only_scope(data_path: str, start: str, end: str) -> None:
    low = str(data_path).lower().replace("\\", "/")
    for bad in ("holdout", "sealed_holdout", "validation"):
        if bad in low:
            raise RunnerSafetyError(f"data_path references forbidden scope: {bad}")
    if REQUIRED_TRAIN_DATA_TOKEN not in low:
        raise RunnerSafetyError(
            f"data_path must point at {REQUIRED_TRAIN_DATA_TOKEN}; got {data_path!r}")
    s, e = _parse_iso(start), _parse_iso(end)
    if s > e:
        raise RunnerSafetyError("start_date after end_date")
    if s >= FIRST_FORBIDDEN_DATE or e >= FIRST_FORBIDDEN_DATE:
        raise RunnerSafetyError("2025/2026 data is forbidden (train-only ends 2024-12-31)")


def assert_safe_request(req: FormalRunRequest) -> None:
    if not req.train_only:
        raise RunnerSafetyError("only train-only runs are permitted in this phase")
    for flag, label in (
        (req.holdout, "holdout"),
        (req.validation, "validation"),
        (req.optimization, "optimization"),
        (req.sweep, "sweep"),
        (req.high_precision, "high_precision"),
        (req.news, "news"),
    ):
        if flag:
            raise RunnerSafetyError(f"forbidden in formal train runner: {label}")
    validate_train_only_scope(req.data_path, req.start_date, req.end_date)


def validate_output_dir(output_dir: str) -> None:
    raw = str(output_dir).strip()
    if not raw:
        raise RunnerSafetyError("empty output_dir")
    p = PurePosixPath(raw.replace("\\", "/"))
    parts = tuple(p.parts)
    if p.suffix.lower() == ".zip" or any(str(x).lower().endswith(".zip") for x in parts):
        raise RunnerSafetyError("ZIP output is forbidden")
    for bad in FORBIDDEN_OUTPUT_PARTS:
        if bad in parts:
            raise RunnerSafetyError(f"output must not write into {bad}/")
    # Project-root / absolute-root style targets.
    if len(parts) <= 1 or (p.is_absolute() and len(parts) <= 2):
        raise RunnerSafetyError("refusing to write at/near project root")
    joined = "/".join(parts)
    if "/".join(REPORTS_AREA_PARTS) not in joined:
        raise RunnerSafetyError(
            f"output_dir must be under {'/'.join(REPORTS_AREA_PARTS)}; got {output_dir!r}")
    if len(parts) <= len(REPORTS_AREA_PARTS):
        raise RunnerSafetyError("refusing to write at formal_train_only root (no run subdir)")


def heavy_output_dir(run_output_dir: str, profile: str) -> str:
    """Heavy artifacts (trades.csv / equity_curve.csv) MUST go here."""
    return str(PurePosixPath(str(run_output_dir).replace("\\", "/")) / HEAVY_SUBDIR / profile)


# ----------------------------------------------------------------------------
# Manifest + reconciliation gate
# ----------------------------------------------------------------------------
def build_run_manifest(
    *,
    run_id: str,
    branch: str,
    commit: str,
    strategy_name: str,
    data_path: str,
    min_timestamp: str,
    max_timestamp: str,
    profiles_run: list[str],
) -> dict[str, Any]:
    if len(set(profiles_run)) != len(profiles_run):
        raise RunnerSafetyError(f"RUN_MANIFEST cannot list duplicate profiles: {profiles_run}")
    if not profiles_run:
        raise RunnerSafetyError("RUN_MANIFEST must record at least one profile actually run")
    return {
        "run_id": run_id,
        "branch": branch,
        "commit": commit,
        "strategy": strategy_name,
        "data_path": data_path,
        "min_timestamp": min_timestamp,
        "max_timestamp": max_timestamp,
        "profiles_run": list(profiles_run),
        "holdout_used": False,
        "validation_run": False,
        "optimization_run": False,
        "sweep_run": False,
        "news_used": False,
        "high_precision_used": False,
        "reconciliation_required": True,
        "train_only": True,
    }


def reconcile_profile_outputs(profile: str, **kwargs: Any) -> dict[str, Any]:
    violations = mr.reconcile_all(**kwargs)
    return {
        "profile": profile,
        "violations": violations,
        "passed": not violations,
    }


def seal_run_only_if_reconciled(
    reconciliations: list[dict[str, Any]],
    manifest: dict[str, Any] | None,
) -> None:
    if manifest is None:
        raise ReconciliationGateError("refusing to seal: no RUN_MANIFEST")
    if not reconciliations:
        raise ReconciliationGateError("refusing to seal: no reconciliation performed")
    bad: set[str] = set()
    for rec in reconciliations:
        for v in rec.get("violations", []):
            bad.add(v["code"])
    if bad:
        raise ReconciliationGateError(f"refusing to seal: violations [{', '.join(sorted(bad))}]")


# ----------------------------------------------------------------------------
# Orchestration (dry-run by default; execute path lazy + gated)
# ----------------------------------------------------------------------------
def preflight(req: FormalRunRequest, base_config: EngineConfig | None = None) -> dict[str, Any]:
    """Validate everything and return a plan WITHOUT running anything."""
    assert_safe_request(req)
    validate_output_dir(req.output_dir)
    configs = build_cost_profile_configs(base_config or EngineConfig(pair="EURUSD"))
    validate_cost_profile_configs(configs)
    manifest = build_run_manifest(
        run_id="PREFLIGHT",
        branch="(unset)",
        commit="(unset)",
        strategy_name=req.strategy_name,
        data_path=req.data_path,
        min_timestamp=req.start_date,
        max_timestamp=req.end_date,
        profiles_run=[name for name, _, _ in PROFILE_PLAN],
    )
    return {
        "mode": "dry_run",
        "executed": False,
        "strategy": req.strategy_name,
        "profiles": {
            name: {"execution_mode": cfg.execution_mode,
                   "cost_profile": resolved_cost_profile(cfg)}
            for name, cfg in configs.items()
        },
        "heavy_output_dirs": {
            name: heavy_output_dir(req.output_dir, name) for name, _, _ in PROFILE_PLAN
        },
        "manifest": manifest,
    }


def run_single_strategy_formal_train_only(
    req: FormalRunRequest,
    base_config: EngineConfig | None = None,
) -> dict[str, Any]:
    """Dry-run by default. Executes a backtest ONLY when `req.execute is True`.

    The execute branch lazy-imports heavy layers so importing this module stays
    side-effect-free; the contract test-suite never invokes it with execute=True.
    """
    plan = preflight(req, base_config)
    if not req.execute:
        return plan

    from research_lab.strategies import STRATEGY_REGISTRY  # lazy
    if req.strategy_name not in STRATEGY_REGISTRY:
        raise RunnerSafetyError(f"unregistered strategy: {req.strategy_name!r}")

    from research_lab.data_loader import load_backtest_data_bundle  # lazy
    from research_lab.engine import run_backtest  # lazy
    from research_lab.report import summarize_result  # lazy

    configs = build_cost_profile_configs(base_config or EngineConfig(pair="EURUSD"))
    validate_cost_profile_configs(configs)
    strategy_module = STRATEGY_REGISTRY[req.strategy_name]
    bundle = load_backtest_data_bundle(
        configs["base"].pair, (req.data_path,), req.start_date, req.end_date,
        "normal_mode", target_timeframe="M1",
    )
    manifest = build_run_manifest(
        run_id=f"{req.strategy_name}_FORMAL",
        branch="(caller-supplied)",
        commit="(caller-supplied)",
        strategy_name=req.strategy_name,
        data_path=req.data_path,
        min_timestamp=req.start_date,
        max_timestamp=req.end_date,
        profiles_run=[n for n, _, _ in PROFILE_PLAN],
    )
    recs: list[dict[str, Any]] = []
    for name, cfg in configs.items():
        result = run_backtest(
            strategy_module=strategy_module, frame=bundle.frame,
            params=strategy_module.DEFAULT_PARAMS, engine_config=cfg,
        )
        summary, trades_exp, _m, _y, equity_exp = summarize_result(
            result.strategy_name, result.trades, result.equity_curve, result.params, False,
        )
        recs.append(reconcile_profile_outputs(
            name,
            trades=trades_exp.to_dict("records"),
            equity_series=list(equity_exp["equity"]),
            starting_equity=float(equity_exp["equity"].iloc[0]),
            profit_factor=float(summary["profit_factor"]),
            expectancy_r=float(summary["expectancy_r"]),
            total_return_pct=float(summary["total_return_pct"]),
            reported_max_drawdown_pct=float(summary["max_drawdown_pct"]),
        ))
    seal_run_only_if_reconciled(recs, manifest)
    return {"mode": "executed", "executed": True, "manifest": manifest, "reconciliations": recs}


# ----------------------------------------------------------------------------
# Fail-closed CLI
# ----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="research_lab.runners.formal_train_runner",
        description="Formal train-only runner (dry-run by default; fail-closed).",
    )
    p.add_argument("--strategy", required=True)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument(
        "--data-path",
        default="05_MARKET_DATA_VAULT/eurusd_data/prepared_train_2015_2024/prepared",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-only", action="store_true", default=True)
    p.add_argument("--execute", action="store_true", default=False)
    # Forbidden flags exist ONLY so the CLI can fail-closed when they are passed.
    p.add_argument("--holdout", action="store_true", default=False)
    p.add_argument("--validation", action="store_true", default=False)
    p.add_argument("--optimize", action="store_true", default=False)
    p.add_argument("--sweep", action="store_true", default=False)
    p.add_argument("--high-precision", action="store_true", default=False)
    p.add_argument("--news", action="store_true", default=False)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    req = FormalRunRequest(
        strategy_name=args.strategy,
        start_date=args.start,
        end_date=args.end,
        data_path=args.data_path,
        output_dir=args.output_dir,
        train_only=True,
        execute=bool(args.execute),
        holdout=bool(args.holdout),
        validation=bool(args.validation),
        optimization=bool(args.optimize),
        sweep=bool(args.sweep),
        high_precision=bool(args.high_precision),
        news=bool(args.news),
    )
    try:
        if not req.execute:
            plan = preflight(req)
            print("[DRY-RUN] preflight OK; aborting before backtest by default.")
            print(plan)
            return 0
        result = run_single_strategy_formal_train_only(req)
        print("[EXECUTED] sealed:", result.get("executed"))
        return 0
    except (RunnerSafetyError, ReconciliationGateError) as exc:
        print(f"[FAIL-CLOSED] {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
