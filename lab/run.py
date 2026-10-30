"""CLI: прогнать стратегию по данным из базы и собрать отчёт.

Пример:
    python -m lab.run --symbol BTCUSDT --tf 4h --strategy donchian \
        --start 2019-01-01 --split 0.7 --report reports/btc_donchian.html
"""

from __future__ import annotations

import argparse
import os
import sys

from . import store
from .collect import TF_MS, parse_time
from .engine import Costs, Settings, buy_and_hold, run_backtest
from .metrics import compute, format_metrics
from .report import write_report
from . import strategies

COST_NOTE = (
    "Комиссия {fee} bp за сторону, слippage {slip} bp, риск {risk}% на сделку, "
    "плечо до {lev}x. Сигнал исполняется на следующем баре."
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Бэктест стратегии по локальной базе")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--tf", default="4h", choices=sorted(TF_MS))
    p.add_argument("--market", default="futures", choices=["spot", "futures"])
    p.add_argument("--strategy", default="donchian", choices=sorted(strategies.REGISTRY))
    p.add_argument("--start", type=parse_time, default=None)
    p.add_argument("--end", type=parse_time, default=None)
    p.add_argument("--split", type=float, default=0.7,
                   help="доля истории для in-sample (0..1), остальное — проверка")
    p.add_argument("--risk", type=float, default=1.0, help="риск на сделку, %%")
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slip-bps", type=float, default=2.0)
    p.add_argument("--equity", type=float, default=1000.0)
    p.add_argument("--short", action="store_true", help="разрешить короткие позиции")
    p.add_argument("--report", default=None, help="куда сохранить HTML-отчёт")
    p.add_argument("--db", default=None)
    p.add_argument("--no-funding", action="store_true", help="не учитывать funding")
    args = p.parse_args(argv)

    con = store.connect(args.db)
    df = store.load_candles(con, args.symbol.upper(), args.market, args.tf,
                            args.start, args.end)
    if df.empty:
        print(f"Нет данных по {args.symbol} {args.market} {args.tf}.\n"
              f"Сначала: python -m lab.collect --symbol {args.symbol} --tf {args.tf} "
              f"--market {args.market}", file=sys.stderr)
        return 1

    funding = None
    if not args.no_funding and args.market == "futures":
        funding = store.load_funding(con, args.symbol.upper(), args.start, args.end)

    sig = strategies.build(args.strategy, df, allow_short=args.short)

    costs = Costs(fee_bps=args.fee_bps, slippage_bps=args.slip_bps)
    settings = Settings(
        initial_equity=args.equity,
        risk_pct=args.risk,
        max_leverage=args.leverage,
        allow_short=args.short,
    )

    res = run_backtest(df, sig.signal, sig.stop_dist, costs, settings, funding)
    bh = buy_and_hold(df, costs, args.equity)

    ppy = 365 * 24 * 3600 * 1000 / TF_MS[args.tf]
    full = compute(res.equity, res.trades, ppy, args.equity, bh)

    split_idx = max(2, min(len(df) - 2, int(len(df) * args.split)))
    split_time = df.index[split_idx]

    is_trades = [t for t in res.trades if t.exit_time <= split_time]
    oos_trades = [t for t in res.trades if t.exit_time > split_time]

    eq = res.equity
    m_is = compute(eq.iloc[:split_idx], is_trades, ppy, args.equity, bh.iloc[:split_idx])
    oos_initial = float(eq.iloc[split_idx - 1])
    m_oos = compute(eq.iloc[split_idx - 1:], oos_trades, ppy, oos_initial,
                    bh.iloc[split_idx - 1:])

    header = (f"{args.symbol.upper()} {args.market} {args.tf} · {args.strategy} · "
              f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d} · {len(df)} баров")
    print(header)
    print("-" * len(header))
    print(format_metrics(full))
    print()
    print(format_metrics(m_is, f"in-sample (до {split_time:%Y-%m-%d})"))
    print()
    print(format_metrics(m_oos, f"out-of-sample (после {split_time:%Y-%m-%d})"))

    # короткий вывод: подтвердился ли край вне выборки
    print()
    if m_oos.trades < 10:
        print(f"ВЫВОД: на проверочном отрезке всего {m_oos.trades} сделок — "
              f"статистики не хватает, нужен больший период.")
    elif m_oos.expectancy_r > 0 and m_oos.profit_factor > 1.1:
        print(f"ВЫВОД: край частично переносится вне выборки "
              f"({m_oos.expectancy_r:+.2f}R, PF {m_oos.profit_factor:.2f}), "
              f"но выборка всё равно мала.")
    else:
        print(f"ВЫВОД: вне выборки край не подтвердился "
              f"({m_oos.expectancy_r:+.2f}R, PF {m_oos.profit_factor:.2f}). "
              f"Скорее всего, параметры подогнаны под историю.")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        write_report(
            args.report,
            symbol=args.symbol.upper(),
            tf=args.tf,
            market=args.market,
            strategy=args.strategy,
            params=strategies.PARAMS[args.strategy],
            full=full,
            insample=m_is,
            oos=m_oos,
            equity=res.equity,
            buy_hold=bh,
            trades=res.trades_frame(),
            costs_note=COST_NOTE.format(
                fee=args.fee_bps, slip=args.slip_bps, risk=args.risk, lev=args.leverage
            ),
            split_at=f"{split_time:%Y-%m-%d}",
        )
        print(f"\nОтчёт: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
