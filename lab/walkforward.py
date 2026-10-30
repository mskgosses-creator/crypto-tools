"""Walk-forward анализ: катящееся окно.

Идея проста и жестока: на каждом шаге мы оптимизируем параметры на прошлом
отрезке (train) и проверяем их на следующем, ещё не виденном (test). Тесты
склеиваются в одну кривую капитала — она и есть честный результат.

Если стратегия живёт только на train и рассыпается на test, значит параметры
подогнаны. Никакая красивая кривая in-sample этого не отменит.

Пример:
    python -m lab.walkforward --symbol BTCUSDT --tf 4h --strategy donchian \
        --train-months 24 --test-months 6 --report reports/wf-donchian.html
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import strategies, store
from .collect import TF_MS, parse_time
from .engine import Costs, Result, Settings, run_backtest
from .metrics import Metrics, compute, format_metrics
from .report import write_wf_report

WARMUP_BARS = 260  # запас истории для индикаторов перед началом отрезка

GRIDS = {
    "donchian": {
        "entry_n": [20, 30, 40, 55, 80],
        "exit_n": [10, 20, 30],
        "stop_atr": [1.5, 2.0, 3.0],
    },
    "sma": {
        "fast": [10, 20, 30, 50],
        "slow": [50, 80, 100, 150, 200],
        "stop_atr": [1.5, 2.0, 3.0],
    },
    "rsi": {
        "period": [7, 14, 21],
        "low": [20, 25, 30, 35],
        "high": [50, 55, 60],
        "stop_atr": [1.5, 2.0, 3.0],
    },
}


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    params: dict = field(default_factory=dict)
    train_metrics: Metrics | None = None
    test_metrics: Metrics | None = None
    base_metrics: Metrics | None = None
    test_equity: pd.Series | None = None
    base_equity: pd.Series | None = None
    test_trades: list = field(default_factory=list)
    base_trades: list = field(default_factory=list)


def make_folds(df: pd.DataFrame, train_months: int, test_months: int,
               step_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Нарезать историю на (начало train, конец train, конец test)."""
    first = df.index[0].normalize()
    last = df.index[-1]
    cursor = first
    out = []
    while True:
        train_end = cursor + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > last:
            break
        out.append((cursor, train_end, test_end))
        cursor = cursor + pd.DateOffset(months=step_months)
    return out


def _segment(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    """Кусок данных с запасом на разогрев индикаторов + индекс начала торговли."""
    i0 = int(df.index.searchsorted(start))
    i1 = int(df.index.searchsorted(end, side="right"))
    lo = max(0, i0 - WARMUP_BARS)
    return df.iloc[lo:i1], i0 - lo


def _run(seg: pd.DataFrame, trade_from: int, name: str, params: dict,
         costs: Costs, settings: Settings, funding: pd.DataFrame | None) -> Result:
    sig = strategies.build(name, seg, allow_short=settings.allow_short, **params)
    return run_backtest(seg, sig.signal, sig.stop_dist, costs, settings, funding)


def _grid(name: str) -> list[dict]:
    grid = GRIDS[name]
    keys = list(grid)
    combos = []
    for values in itertools.product(*(grid[k] for k in keys)):
        combo = dict(zip(keys, values))
        if name == "sma" and combo["fast"] >= combo["slow"]:
            continue
        if name == "rsi" and combo["low"] >= combo["high"]:
            continue
        combos.append(combo)
    return combos


def _score(m: Metrics, min_trades: int) -> tuple[float, float]:
    """Цель оптимизации: ожидание в R, при достаточном числе сделок."""
    if m.trades < min_trades:
        return (-1e9, -1e9)
    return (m.expectancy_r, m.profit_factor if m.profit_factor != float("inf") else 99.0)


def run_walkforward(
    df: pd.DataFrame,
    name: str,
    *,
    train_months: int = 24,
    test_months: int = 6,
    step_months: int = 6,
    costs: Costs | None = None,
    settings: Settings | None = None,
    funding: pd.DataFrame | None = None,
    min_trades: int = 8,
    periods_per_year: float = 2190.0,
    verbose: bool = True,
) -> tuple[list[Fold], pd.Series, list, pd.Series, list]:
    """Прогнать walk-forward и вернуть склейку out-of-sample."""
    if costs is None:
        costs = Costs()
    if settings is None:
        settings = Settings()

    folds_spec = make_folds(df, train_months, test_months, step_months)
    if not folds_spec:
        raise ValueError("истории не хватает на одно окно: уменьшите train/test months")

    combos = _grid(name)
    defaults = strategies.PARAMS[name]
    result: list[Fold] = []
    equity_parts: list[pd.Series] = []
    base_parts: list[pd.Series] = []
    all_trades: list = []
    base_trades_all: list = []

    equity = float(settings.initial_equity)
    base_equity = float(settings.initial_equity)

    t0 = time.time()
    for k, (tr_s, tr_e, te_end) in enumerate(folds_spec, start=1):
        fold = Fold(train_start=tr_s, train_end=tr_e, test_start=tr_e, test_end=te_end)

        # --- оптимизация на train ---
        train_seg, train_from = _segment(df, tr_s, tr_e)
        best, best_score = None, (-1e9, -1e9)
        for params in combos:
            res = _run(train_seg, train_from, name, params, costs,
                       Settings(initial_equity=settings.initial_equity,
                                risk_pct=settings.risk_pct,
                                max_leverage=settings.max_leverage,
                                allow_short=settings.allow_short,
                                trade_from=train_from),
                       funding)
            m = compute(res.equity, res.trades, periods_per_year, settings.initial_equity)
            sc = _score(m, min_trades)
            if sc > best_score:
                best_score, best = sc, (params, m)

        if best is None or best_score[0] <= -1e8:
            # на train не набралось сделок — берём параметры по умолчанию
            params = dict(defaults)
            train_metrics = None
        else:
            params, train_metrics = best

        # --- проверка на test ---
        test_seg, test_from = _segment(df, tr_e, te_end)
        res_test = _run(test_seg, test_from, name, params, costs,
                        Settings(initial_equity=equity,
                                 risk_pct=settings.risk_pct,
                                 max_leverage=settings.max_leverage,
                                 allow_short=settings.allow_short,
                                 trade_from=test_from),
                        funding)
        test_eq = res_test.equity.iloc[test_from:]
        if len(test_eq) == 0:
            continue
        equity = float(test_eq.iloc[-1])
        equity_parts.append(test_eq)
        all_trades.extend(res_test.trades)

        # --- базовая линия: параметры по умолчанию на том же test ---
        res_base = _run(test_seg, test_from, name, dict(defaults), costs,
                        Settings(initial_equity=base_equity,
                                 risk_pct=settings.risk_pct,
                                 max_leverage=settings.max_leverage,
                                 allow_short=settings.allow_short,
                                 trade_from=test_from),
                        funding)
        base_eq = res_base.equity.iloc[test_from:]
        base_equity = float(base_eq.iloc[-1])
        base_parts.append(base_eq)
        base_trades_all.extend(res_base.trades)

        fold.params = params
        fold.train_metrics = train_metrics
        fold.test_metrics = compute(test_eq, res_test.trades, periods_per_year, float(test_eq.iloc[0]))
        fold.base_metrics = compute(base_eq, res_base.trades, periods_per_year, float(base_eq.iloc[0]))
        fold.test_equity = test_eq
        fold.base_equity = base_eq
        fold.test_trades = res_test.trades
        fold.base_trades = res_base.trades
        result.append(fold)

        if verbose:
            p = " ".join(f"{k}={v}" for k, v in params.items())
            tm = fold.test_metrics
            print(f"  окно {k:2d}: train {tr_s:%Y-%m}..{tr_e:%Y-%m} "
                  f"test {tr_e:%Y-%m}..{te_end:%Y-%m} | {p:38s} | "
                  f"test {tm.expectancy_r:+.2f}R, {tm.trades} сд.")

    if verbose:
        print(f"  всего {len(result)} окон за {time.time() - t0:.1f}s")

    agg = pd.concat(equity_parts) if equity_parts else pd.Series(dtype=float)
    return result, agg, all_trades, pd.concat(base_parts) if base_parts else pd.Series(dtype=float), base_trades_all


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward анализ стратегии")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--tf", default="4h", choices=sorted(TF_MS))
    p.add_argument("--market", default="futures", choices=["spot", "futures"])
    p.add_argument("--strategy", default="donchian", choices=sorted(GRIDS))
    p.add_argument("--train-months", type=int, default=24)
    p.add_argument("--test-months", type=int, default=6)
    p.add_argument("--step-months", type=int, default=6)
    p.add_argument("--min-trades", type=int, default=8)
    p.add_argument("--risk", type=float, default=1.0)
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slip-bps", type=float, default=2.0)
    p.add_argument("--equity", type=float, default=1000.0)
    p.add_argument("--short", action="store_true")
    p.add_argument("--report", default=None)
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)

    con = store.connect(args.db)
    df = store.load_candles(con, args.symbol.upper(), args.market, args.tf)
    if df.empty:
        print("Нет данных. Сначала: python -m lab.collect ...", file=sys.stderr)
        return 1

    funding = None
    if args.market == "futures":
        funding = store.load_funding(con, args.symbol.upper())

    costs = Costs(fee_bps=args.fee_bps, slippage_bps=args.slip_bps)
    settings = Settings(initial_equity=args.equity, risk_pct=args.risk,
                        max_leverage=args.leverage, allow_short=args.short)

    print(f"walk-forward: {args.symbol.upper()} {args.market} {args.tf} · {args.strategy}")
    print(f"окно: train {args.train_months} мес, test {args.test_months} мес, "
          f"шаг {args.step_months} мес, риск {args.risk}%")

    folds, agg, trades, base_agg, base_trades = run_walkforward(
        df, args.strategy, train_months=args.train_months,
        test_months=args.test_months, step_months=args.step_months,
        costs=costs, settings=settings, funding=funding,
        min_trades=args.min_trades,
    )

    if not folds:
        print("Ни одного окна не получилось.", file=sys.stderr)
        return 1

    ppy = 365 * 24 * 3600 * 1000 / TF_MS[args.tf]
    m_agg = compute(agg, trades, ppy, args.equity)
    m_base = compute(base_agg, base_trades, ppy, args.equity)

    print()
    print(format_metrics(m_agg, "OUT-OF-SAMPLE (склейка всех окон)"))
    print()
    print(format_metrics(m_base, "для сравнения: те же окна с параметрами по умолчанию"))

    wins = sum(1 for f in folds if f.test_metrics and f.test_metrics.expectancy_r > 0)
    print()
    print(f"прибыльных окон: {wins} из {len(folds)}")
    if m_agg.expectancy_r <= 0:
        verdict = ("край не подтверждается: на невидимых данных стратегия "
                   "не зарабатывает стабильно")
    elif m_base.expectancy_r <= 0:
        verdict = ("край есть, но только с подбором параметров: параметры "
                   "по умолчанию на тех же окнах убыточны")
    elif m_agg.expectancy_r >= m_base.expectancy_r:
        verdict = ("край подтверждается, и подбор параметров не хуже базовой "
                   "линии")
    else:
        verdict = ("край подтверждается, но подбор параметров не помогает: "
                   "параметры по умолчанию на тех же окнах отработали лучше. "
                   "Оптимизация — источник шума, а не прибыли")
    print("ВЫВОД:", verdict)

    if args.report:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        write_wf_report(
            args.report,
            symbol=args.symbol.upper(), tf=args.tf, market=args.market,
            strategy=args.strategy, folds=folds,
            full=m_agg, base=m_base, equity=agg, base_equity=base_agg,
            trades=trades, train_months=args.train_months,
            test_months=args.test_months, step_months=args.step_months,
            risk=args.risk, fee=args.fee_bps, slip=args.slip_bps,
        )
        print(f"\nОтчёт: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


