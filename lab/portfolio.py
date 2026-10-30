"""Портфель из N инструментов с равным риском.

Зачем это нужно: край одной пары легко оказывается случайностью — одна
история, один режим рынка. Если то же правило выживает на двадцати разных
монетах, это уже статистика, а не совпадение.

Модель простая и честная:

* Капитал делится поровну между инструментами (sleeve).
* В каждом sleeve — тот же движок, что и в одиночном бэктесте: стоп, комиссии,
  слippage, funding, риск 1% от капитала sleeve на сделку.
* Портфель — сумма sleeve-кривых. Ребалансировки нет: каждый sleeve
  компаундится независимо.

Чего модель не делает (важно для интерпретации):

* Не учитывает корреляцию: в кризис все лонги падают вместе, и просадка
  портфеля будет глубже, чем намекает сумма независимых долей.
* Не ограничивает общий нотional: 20 одновременных позиций могут дать
  суммарное плечо больше единицы. Это видно в отчёте по экспозиции.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import strategies, store
from .collect import TF_MS, parse_time
from .engine import Costs, Result, Settings, buy_and_hold, run_backtest
from .metrics import Metrics, compute, format_metrics
from .report import CSS, _kpi, _svg_curve

DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
    "BCHUSDT", "ETCUSDT", "XLMUSDT", "TRXUSDT", "EOSUSDT",
    "ATOMUSDT", "NEOUSDT", "DASHUSDT", "ZECUSDT", "XTZUSDT",
]


@dataclass
class Sleeve:
    symbol: str
    equity: pd.Series
    trades: list
    metrics: Metrics
    buy_hold: pd.Series


def trade_stats(trades: list) -> dict:
    """Статистика по объединённым сделкам (в R — размер капитала не важен)."""
    if not trades:
        return {
            "count": 0, "winrate": 0.0, "expectancy_r": 0.0, "pf": 0.0,
            "avg_win_r": 0.0, "avg_loss_r": 0.0, "max_streak": 0, "avg_bars": 0.0,
        }
    rs = np.array([t.r_multiple for t in trades], dtype=float)
    wins, losses = rs[rs > 0], rs[rs <= 0]
    gross_win, gross_loss = float(wins.sum()), float(abs(losses.sum()))
    streak = max_streak = 0
    for r in rs:
        if r <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return {
        "count": len(rs),
        "winrate": len(wins) / len(rs) * 100,
        "expectancy_r": float(rs.mean()),
        "pf": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "avg_win_r": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss_r": float(losses.mean()) if len(losses) else 0.0,
        "max_streak": max_streak,
        "avg_bars": float(np.mean([t.bars for t in trades])),
    }


def run_portfolio(
    dfs: dict[str, pd.DataFrame],
    strategy: str,
    *,
    start: pd.Timestamp | None = None,
    costs: Costs | None = None,
    risk_pct: float = 1.0,
    max_leverage: float = 1.0,
    allow_short: bool = False,
    funding: dict[str, pd.DataFrame] | None = None,
    total_equity: float = 10_000.0,
    periods_per_year: float = 2190.0,
) -> tuple[pd.Series, pd.Series, list[Sleeve], list]:
    """Прогнать стратегию по каждому инструменту и сложить sleeve-кривые."""
    if costs is None:
        costs = Costs()

    n = len(dfs)
    per = total_equity / n
    sleeves: list[Sleeve] = []
    curves: dict[str, pd.Series] = {}
    bh_curves: dict[str, pd.Series] = {}
    all_trades: list = []

    for sym, df in dfs.items():
        i0 = int(df.index.searchsorted(start)) if start is not None else 0
        if i0 >= len(df) - 20:
            continue

        sig = strategies.build(strategy, df, allow_short=allow_short)
        st = Settings(
            initial_equity=per,
            risk_pct=risk_pct,
            max_leverage=max_leverage,
            allow_short=allow_short,
            trade_from=i0,
        )
        res: Result = run_backtest(
            df, sig.signal, sig.stop_dist, costs, st,
            (funding or {}).get(sym),
        )
        eq = res.equity.iloc[i0:]
        if len(eq) == 0:
            continue

        bh = buy_and_hold(df.iloc[i0:], costs, per)
        curves[sym] = eq
        bh_curves[sym] = bh
        all_trades.extend(res.trades)
        sleeves.append(
            Sleeve(
                symbol=sym,
                equity=eq,
                trades=res.trades,
                metrics=compute(eq, res.trades, periods_per_year, per, bh),
                buy_hold=bh,
            )
        )

    if not curves:
        empty = pd.Series(dtype=float)
        return empty, empty, [], []

    idx = sorted(set().union(*[c.index for c in curves.values()]))
    frame = pd.DataFrame({s: c.reindex(idx).ffill() for s, c in curves.items()})
    frame = frame.fillna(per)
    portfolio = frame.sum(axis=1).rename("portfolio")

    bh_frame = pd.DataFrame({s: c.reindex(idx).ffill() for s, c in bh_curves.items()})
    bh_portfolio = bh_frame.fillna(per).sum(axis=1).rename("buy_hold")

    return portfolio, bh_portfolio, sleeves, all_trades


CSS_EXTRA = """
.sleeves td{font-size:12.5px}
.bar{height:6px;background:#242d38;border-radius:3px;overflow:hidden;min-width:60px}
.bar span{display:block;height:100%}
"""


def write_portfolio_report(
    path: str,
    *,
    strategy: str,
    tf: str,
    market: str,
    sleeves: list[Sleeve],
    portfolio: pd.Series,
    buy_hold: pd.Series,
    trades: list,
    metrics: Metrics,
    start: pd.Timestamp,
    end: pd.Timestamp,
    risk_pct: float,
    capital: float,
    fee_bps: float,
    slip_bps: float,
) -> None:
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ts = trade_stats(trades)
    pf = "∞" if ts["pf"] == float("inf") else f"{ts['pf']:.2f}"
    positive = sum(1 for s in sleeves if s.metrics.total_return > 0)

    rows = []
    for s in sorted(sleeves, key=lambda x: -x.metrics.total_return):
        cls = "win" if s.metrics.total_return >= 0 else "loss"
        rows.append(
            "<tr>"
            f"<td><b>{s.symbol.replace('USDT', '')}</b></td>"
            f"<td class='num {cls}'>{s.metrics.total_return:+.1f}%</td>"
            f"<td class='num'>{s.metrics.trades}</td>"
            f"<td class='num'>{s.metrics.winrate:.0f}%</td>"
            f"<td class='num'>{s.metrics.expectancy_r:+.2f}R</td>"
            f"<td class='num'>{s.metrics.max_dd_pct:.0f}%</td>"
            f"<td class='num'>{s.metrics.buy_hold:+.1f}%</td>"
            "</tr>"
        )

    doc = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Портфель · {strategy} · {tf}</title>
<style>{CSS}{CSS_EXTRA}</style></head>
<body><div class="wrap">
  <h1>Портфель · {len(sleeves)} инструментов · {strategy}</h1>
  <div class="sub">{market}, {tf} · {start:%Y-%m-%d} → {end:%Y-%m-%d}
   · капитал {_fmt(capital)} на {len(sleeves)} равных долей
   · риск {risk_pct}% от доли на сделку
   · комиссия {fee_bps} bp, слippage {slip_bps} bp
   · отчёт от {gen}</div>

  <div class="warn">
    Доли независимы: каждая компаундится сама по себе, ребалансировки нет.
    Корреляция не моделируется — в реальном кризисе все лонги падают вместе,
    и просадка будет глубже.
  </div>

  <div class="grid" style="margin-bottom:16px">
    {_kpi("Итог портфеля", f"{metrics.total_return:+.1f}%", f"{_fmt(metrics.initial)} → {_fmt(metrics.final)}", "win" if metrics.total_return >= 0 else "loss")}
    {_kpi("Макс. просадка", f"{metrics.max_dd_pct:.1f}%", _fmt(metrics.max_dd))}
    {_kpi("CAGR", f"{metrics.cagr:+.1f}%", "в год")}
    {_kpi("Сделок", str(ts["count"]), f"winrate {ts['winrate']:.0f}%")}
    {_kpi("Ожидание", f"{ts['expectancy_r']:+.2f}R", "по всем сделкам", "win" if ts["expectancy_r"] > 0 else "loss")}
    {_kpi("Profit factor", pf, "по R, не по деньгам")}
    {_kpi("Buy & hold", f"{metrics.buy_hold:+.0f}%", "равновесный портфель тех же монет", "win" if metrics.buy_hold >= 0 else "loss")}
    {_kpi("Прибыльных монет", f"{positive} / {len(sleeves)}", "итог > 0")}
    {_kpi("Sharpe", f"{metrics.sharpe:.2f}", f"экспозиция {metrics.exposure:.0f}%")}
  </div>

  <div class="card">
    <h2>Кривая портфеля</h2>
    {_svg_curve(portfolio)}
    <p style="color:#8a95a3;font-size:12px;margin:10px 0 0">
      Только портфель: buy &amp; hold тех же монет растёт в разы быстрее
      и на общей оси превратил бы эту линию в прямую. Число — в KPI выше
      и в колонке таблицы.
    </p>
  </div>

  <div class="card">
    <h2>По инструментам</h2>
    <table class="sleeves"><thead><tr>
      <th>Инструмент</th><th class="num">Итог</th><th class="num">Сделок</th>
      <th class="num">Winrate</th><th class="num">Ожидание</th>
      <th class="num">Просадка</th><th class="num">Buy &amp; hold</th>
    </tr></thead><tbody>{''.join(rows)}</tbody></table>
  </div>

  <div class="card">
    <h2>Что это значит</h2>
    <p style="color:#b7c1cd;font-size:13px;line-height:1.65">
      Если край реален, он должен проявляться на большинстве инструментов,
      а не на одном. Разброс между строками таблицы — это и есть оценка
      устойчивости: когда половина монет в плюсе, а половина в минусе,
      результат портфеля почти целиком определяется везением.
      Сравнение с buy &amp; hold показывает, платите ли вы за управление
      риском меньшей доходностью — и насколько меньшей.
    </p>
  </div>

  <div class="foot">
    Комиссия и слippage на каждой стороне, funding по фактическим ставкам.
    Сигнал исполняется на следующем баре, стоп проверяется внутри бара.
    <br>Не инвестиционная рекомендация.
  </div>
</div></body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def _fmt(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Портфельный бэктест: равный риск на N инструментов")
    p.add_argument("--symbols", nargs="+", default=None,
                   help="по умолчанию 20 ликвидных перпетуалов")
    p.add_argument("--strategy", default="donchian", choices=sorted(strategies.REGISTRY))
    p.add_argument("--tf", default="4h", choices=sorted(TF_MS))
    p.add_argument("--market", default="futures", choices=["spot", "futures"])
    p.add_argument("--start", type=parse_time, default=parse_time("2021-01-01"))
    p.add_argument("--capital", type=float, default=10_000.0)
    p.add_argument("--risk", type=float, default=1.0, help="риск на сделку, %% от доли")
    p.add_argument("--leverage", type=float, default=1.0)
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slip-bps", type=float, default=2.0)
    p.add_argument("--short", action="store_true")
    p.add_argument("--report", default=None)
    p.add_argument("--db", default=None)
    args = p.parse_args(argv)

    con = store.connect(args.db)
    symbols = [s.upper() for s in (args.symbols or DEFAULT_UNIVERSE)]
    start = pd.Timestamp(args.start, unit="ms", tz="UTC")

    dfs: dict[str, pd.DataFrame] = {}
    funding: dict[str, pd.DataFrame] = {}
    missing = []
    for sym in symbols:
        df = store.load_candles(con, sym, args.market, args.tf)
        if df.empty or df.index[0] > start + pd.Timedelta(days=30):
            missing.append(sym)
            continue
        dfs[sym] = df
        if args.market == "futures":
            funding[sym] = store.load_funding(con, sym)

    if missing:
        print(f"пропущены (нет данных с {start:%Y-%m}): {', '.join(missing)}")
    if not dfs:
        print("Нет данных. Сначала: python -m lab.collect --symbol ... --tf 4h", file=sys.stderr)
        return 1

    print(f"портфель: {len(dfs)} инструментов · {args.strategy} · {args.tf} · "
          f"с {start:%Y-%m-%d}")

    costs = Costs(fee_bps=args.fee_bps, slippage_bps=args.slip_bps)
    ppy = 365 * 24 * 3600 * 1000 / TF_MS[args.tf]

    portfolio, bh, sleeves, trades = run_portfolio(
        dfs, args.strategy, start=start, costs=costs, risk_pct=args.risk,
        max_leverage=args.leverage, allow_short=args.short,
        funding=funding, total_equity=args.capital, periods_per_year=ppy,
    )
    if portfolio.empty:
        print("Не удалось собрать портфель.", file=sys.stderr)
        return 1

    m = compute(portfolio, trades, ppy, args.capital, bh)
    m.exposure = float(np.mean([s.metrics.exposure for s in sleeves])) if sleeves else 0.0
    ts = trade_stats(trades)

    print()
    print(format_metrics(m, f"ПОРТФЕЛЬ ({len(sleeves)} инструментов, равный риск)"))
    print(f"сделок всего   {ts['count']}, winrate {ts['winrate']:.1f}%, "
          f"ожидание {ts['expectancy_r']:+.3f}R, PF {ts['pf']:.2f}")
    print(f"серия убытков  {ts['max_streak']} подряд")
    positive = sum(1 for s in sleeves if s.metrics.total_return > 0)
    print(f"прибыльных монет: {positive} из {len(sleeves)}")

    print()
    print("по инструментам:")
    for s in sorted(sleeves, key=lambda x: -x.metrics.total_return):
        print(f"  {s.symbol:10s} {s.metrics.total_return:+7.1f}%  "
              f"сделок {s.metrics.trades:3d}  winrate {s.metrics.winrate:4.0f}%  "
              f"ожидание {s.metrics.expectancy_r:+.2f}R  "
              f"просадка {s.metrics.max_dd_pct:4.0f}%")

    print()
    if ts["expectancy_r"] > 0 and positive >= len(sleeves) * 0.6:
        print("ВЫВОД: край подтверждается на большинстве инструментов — это уже не совпадение.")
    elif ts["expectancy_r"] > 0:
        print("ВЫВОД: портфель в плюсе, но прибыль держится на меньшинстве монет — хрупко.")
    else:
        print("ВЫВОД: край не переносится на широкий рынок.")

    if args.report:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        write_portfolio_report(
            args.report, strategy=args.strategy, tf=args.tf, market=args.market,
            sleeves=sleeves, portfolio=portfolio, buy_hold=bh, trades=trades,
            metrics=m, start=portfolio.index[0], end=portfolio.index[-1],
            risk_pct=args.risk, capital=args.capital,
            fee_bps=args.fee_bps, slip_bps=args.slip_bps,
        )
        print(f"\nОтчёт: {args.report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())



