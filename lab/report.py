"""HTML-отчёт: метрики, кривая капитала, сделки.

Отчёт самодостаточный (один файл, без внешних ресурсов) — можно отправить
кому угодно или положить в репозиторий.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .metrics import Metrics

CSS = """
:root{--bg:#0e1115;--panel:#151a21;--line:#242d38;--text:#dbe2ea;--muted:#8a95a3;
--gold:#c9a227;--green:#4ea87a;--red:#cf5f5f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:28px 22px 60px}
h1{font-size:20px;margin:0 0 4px;font-weight:650}
.sub{color:var(--muted);font-size:13px;margin-bottom:22px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;
padding:16px 18px;margin-bottom:16px}
h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);
margin:0 0 14px;font-weight:600}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi .k{font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.kpi .v{font-size:20px;font-weight:650;margin-top:6px;font-variant-numeric:tabular-nums}
.kpi .d{font-size:11.5px;color:var(--muted);margin-top:4px}
.win{color:var(--green)}.loss{color:var(--red)}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);padding:7px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:7px 8px;border-bottom:1px solid #1c242e;white-space:nowrap}
.num{text-align:right;font-variant-numeric:tabular-nums}
tr:hover td{background:#171e26}
.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.foot{color:#5d6673;font-size:11.5px;margin-top:24px;line-height:1.7}
.warn{background:rgba(201,162,39,.07);border-left:3px solid var(--gold);
padding:10px 14px;border-radius:0 8px 8px 0;font-size:12.5px;margin-bottom:16px}
@media(max-width:760px){.two{grid-template-columns:1fr}}
.split .kpi{padding:10px 12px}
.split .kpi .v{font-size:17px}
.split .grid{grid-template-columns:repeat(2,minmax(0,1fr))}
"""


def _fmt_money(v: float, d: int = 0) -> str:
    return f"{v:,.{d}f}".replace(",", " ")


def _svg_curve(equity: pd.Series, buy_hold: pd.Series | None = None,
               width: int = 1060, height: int = 300) -> str:
    """Кривая капитала в SVG, с заливкой и базовой линией buy & hold."""
    eq = equity.dropna()
    if len(eq) < 2:
        return "<p style='color:#8a95a3'>Недостаточно данных для графика.</p>"

    step = max(1, len(eq) // 1800)
    ys = eq.iloc[::step]
    bh = buy_hold.dropna().iloc[::step] if buy_hold is not None and len(buy_hold) else None

    pad_l, pad_r, pad_t, pad_b = 64, 16, 14, 24
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b

    values = list(ys.values)
    if bh is not None:
        values += list(bh.values)
    lo, hi = float(min(values)), float(max(values))
    if hi == lo:
        hi += 1
        lo -= 1

    # при большом разбросе (buy & hold в разы выше) лог-шкала честнее:
    # линейная превращает стратегию в плоскую линию и врёт о масштабе
    log_scale = lo > 0 and (hi / lo) > 8
    if log_scale:
        lo = lo / 1.15
        hi = hi * 1.15
    else:
        margin = (hi - lo) * 0.1
        lo -= margin
        hi += margin

    def x(i: int, n: int) -> float:
        return pad_l + iw * i / (n - 1)

    if log_scale:
        llo, lhi = np.log(lo), np.log(hi)

        def y(v: float) -> float:
            return pad_t + ih * (1 - (np.log(v) - llo) / (lhi - llo))
    else:
        def y(v: float) -> float:
            return pad_t + ih * (1 - (v - lo) / (hi - lo))

    n = len(ys)
    pts = " ".join(f"{x(i, n):.1f},{y(v):.1f}" for i, v in enumerate(ys.values))
    area = f"{pad_l},{pad_t + ih} {pts} {x(n - 1, n):.1f},{pad_t + ih}"

    up = ys.iloc[-1] >= ys.iloc[0]
    color = "#4ea87a" if up else "#cf5f5f"

    grid = []
    for k in range(5):
        v = float(np.exp(llo + (lhi - llo) * k / 4)) if log_scale else lo + (hi - lo) * k / 4
        yy = y(v)
        grid.append(f"<line x1='{pad_l}' y1='{yy:.1f}' x2='{width - pad_r}' y2='{yy:.1f}' "
                    f"stroke='#202934' stroke-width='1'/>")
        grid.append(f"<text x='{pad_l - 8}' y='{yy + 4:.1f}' fill='#78838f' font-size='10' "
                    f"text-anchor='end'>{_fmt_money(v)}</text>")

    bh_line = ""
    if bh is not None:
        bpts = " ".join(f"{x(i, len(bh)):.1f},{y(v):.1f}" for i, v in enumerate(bh.values))
        bh_line = (f"<polyline points='{bpts}' fill='none' stroke='#6b7787' "
                   f"stroke-width='1.2' stroke-dasharray='4 4' opacity='.8'/>")

    return (
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' "
        f"xmlns='http://www.w3.org/2000/svg'>"
        f"<defs><linearGradient id='g' x1='0' y1='0' x2='0' y2='1'>"
        f"<stop offset='0%' stop-color='{color}' stop-opacity='.22'/>"
        f"<stop offset='100%' stop-color='{color}' stop-opacity='0'/>"
        f"</linearGradient></defs>"
        + "".join(grid)
        + f"<polygon points='{area}' fill='url(#g)'/>"
        + bh_line
        + f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='2' "
          f"stroke-linejoin='round'/>"
        + f"<circle cx='{x(n - 1, n):.1f}' cy='{y(ys.iloc[-1]):.1f}' r='3.5' fill='{color}'/>"
        + "</svg>"
    )


def _kpi(label: str, value: str, detail: str = "", cls: str = "") -> str:
    return (
        f"<div class='kpi'><div class='k'>{html.escape(label)}</div>"
        f"<div class='v {cls}'>{value}</div>"
        f"<div class='d'>{html.escape(detail)}</div></div>"
    )


def _metrics_block(m: Metrics) -> str:
    pf = "∞" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    return "".join([
        _kpi("Итог", f"{m.total_return:+.1f}%",
             f"{_fmt_money(m.initial)} → {_fmt_money(m.final)}",
             "win" if m.total_return >= 0 else "loss"),
        _kpi("Макс. просадка", f"{m.max_dd_pct:.1f}%", f"{_fmt_money(m.max_dd)}"),
        _kpi("Сделок", str(m.trades), f"winrate {m.winrate:.0f}%"),
        _kpi("Ожидание", f"{m.expectancy_r:+.2f}R", "на сделку",
             "win" if m.expectancy_r > 0 else "loss"),
        _kpi("Profit factor", pf, "прибыль / убыток"),
        _kpi("Sharpe", f"{m.sharpe:.2f}", f"в позиции {m.exposure:.0f}%"),
    ])


def _trades_table(trades: pd.DataFrame, limit: int = 40) -> str:
    if trades.empty:
        return "<p style='color:#8a95a3'>Сделок не было.</p>"
    t = trades.tail(limit).copy()
    rows = []
    for _, r in t.iterrows():
        cls = "win" if r["pnl"] > 0 else "loss"
        rows.append(
            "<tr>"
            f"<td>{r['entry_time']:%Y-%m-%d %H:%M}</td>"
            f"<td>{r['side']}</td>"
            f"<td class='num'>{_fmt_money(r['entry_price'], 2)}</td>"
            f"<td class='num'>{_fmt_money(r['exit_price'], 2)}</td>"
            f"<td class='num {cls}'>{r['pnl']:+,.2f}</td>"
            f"<td class='num'>{r['r']:+.2f}R</td>"
            f"<td>{r['reason']}</td>"
            f"<td class='num'>{int(r['bars'])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Вход</th><th>Сторона</th><th class='num'>Цена входа</th>"
        "<th class='num'>Выход</th><th class='num'>PnL</th><th class='num'>R</th>"
        "<th>Причина</th><th class='num'>Баров</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def write_report(
    path: str,
    *,
    symbol: str,
    tf: str,
    market: str,
    strategy: str,
    params: dict,
    full: Metrics,
    insample: Metrics | None,
    oos: Metrics | None,
    equity: pd.Series,
    buy_hold: pd.Series | None,
    trades: pd.DataFrame,
    costs_note: str,
    split_at: str | None = None,
) -> None:
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())

    split_block = ""
    if insample and oos:
        split_block = f"""
        <div class="card split">
          <h2>In-sample / out-of-sample</h2>
          <div class="two">
            <div>
              <div class="sub" style="margin-bottom:10px">Обучение (до {split_at})</div>
              <div class="grid">{_metrics_block(insample)}</div>
            </div>
            <div>
              <div class="sub" style="margin-bottom:10px">Проверка (после {split_at})</div>
              <div class="grid">{_metrics_block(oos)}</div>
            </div>
          </div>
          <p style="color:#8a95a3;font-size:12.5px;margin:14px 0 0">
            Вторая половина не участвовала в выборе параметров. Если результаты
            расходятся в разы — стратегия подогнана под историю.
          </p>
        </div>"""

    doc = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(symbol)} {html.escape(strategy)} — бэктест</title>
<style>{CSS}</style></head>
<body><div class="wrap">
  <h1>{html.escape(symbol)} · {html.escape(strategy)} · {html.escape(tf)}</h1>
  <div class="sub">{html.escape(market)}, параметры: {html.escape(param_str)}
   · отчёт от {gen}</div>

  <div class="warn">{html.escape(costs_note)}</div>

  <div class="grid" style="margin-bottom:16px">{_metrics_block(full)}</div>

  <div class="card">
    <h2>Кривая капитала</h2>
    {_svg_curve(equity, buy_hold)}
    <p style="color:#8a95a3;font-size:12px;margin:10px 0 0">
      Пунктир — buy &amp; hold за тот же период.
    </p>
  </div>

  {split_block}

  <div class="card">
    <h2>Сделки (последние {min(len(trades), 40)})</h2>
    {_trades_table(trades)}
  </div>

  <div class="foot">
    Комиссия и слippage учтены на каждой стороне, funding — по фактическим ставкам.
    Сигнал исполняется на следующем баре после закрытия, стоп проверяется внутри бара.
    <br>Не инвестиционная рекомендация.
  </div>
</div></body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def write_wf_report(
    path: str,
    *,
    symbol: str,
    tf: str,
    market: str,
    strategy: str,
    folds: list,
    full: Metrics,
    base: Metrics,
    equity: pd.Series,
    base_equity: pd.Series,
    trades: list,
    train_months: int,
    test_months: int,
    step_months: int,
    risk: float,
    fee: float,
    slip: float,
) -> None:
    """Отчёт walk-forward: окна, выбранные параметры, склейка out-of-sample."""
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    pf = "∞" if full.profit_factor == float("inf") else f"{full.profit_factor:.2f}"
    base_pf = "∞" if base.profit_factor == float("inf") else f"{base.profit_factor:.2f}"
    wins = sum(1 for f in folds if f.test_metrics and f.test_metrics.expectancy_r > 0)

    rows = []
    for i, f in enumerate(folds, start=1):
        p = " ".join(f"{k}={v}" for k, v in f.params.items())
        tm, trm = f.test_metrics, f.train_metrics
        train_r = f"{trm.expectancy_r:+.2f}R" if trm else "—"
        test_r = f"{tm.expectancy_r:+.2f}R" if tm else "—"
        cls = "win" if tm and tm.expectancy_r > 0 else "loss"
        rows.append(
            "<tr>"
            f"<td class='num'>{i}</td>"
            f"<td>{f.train_start:%Y-%m} → {f.train_end:%Y-%m}</td>"
            f"<td>{f.test_start:%Y-%m} → {f.test_end:%Y-%m}</td>"
            f"<td style='color:#8a95a3;font-size:11.5px'>{html.escape(p)}</td>"
            f"<td class='num'>{train_r}</td>"
            f"<td class='num {cls}'>{test_r}</td>"
            f"<td class='num'>{tm.trades if tm else 0}</td>"
            f"<td class='num'>{(tm.total_return if tm else 0):+.1f}%</td>"
            "</tr>"
        )

    fold_table = (
        "<table><thead><tr><th class='num'>#</th><th>Обучение</th><th>Проверка</th>"
        "<th>Выбранные параметры</th><th class='num'>Train R</th>"
        "<th class='num'>Test R</th><th class='num'>Сделок</th>"
        "<th class='num'>Доход</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )

    doc = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(symbol)} {html.escape(strategy)} — walk-forward</title>
<style>{CSS}</style></head>
<body><div class="wrap">
  <h1>{html.escape(symbol)} · {html.escape(strategy)} · walk-forward</h1>
  <div class="sub">{html.escape(market)}, {html.escape(tf)}
   · train {train_months} мес → test {test_months} мес, шаг {step_months} мес
   · риск {risk}%, комиссия {fee} bp, слippage {slip} bp
   · {len(folds)} окон · отчёт от {gen}</div>

  <div class="warn">
    Каждое окно оптимизировалось вслепую на прошлом и проверялось на следующем
    отрезке. Кривая ниже — склейка именно проверочных отрезков, где параметры
    не подбирались.
  </div>

  <div class="grid" style="margin-bottom:16px">
    {_kpi("Итог OOS", f"{full.total_return:+.1f}%", f"{_fmt_money(full.initial)} → {_fmt_money(full.final)}", "win" if full.total_return >= 0 else "loss")}
    {_kpi("Макс. просадка", f"{full.max_dd_pct:.1f}%", f"{_fmt_money(full.max_dd)}")}
    {_kpi("Сделок", str(full.trades), f"winrate {full.winrate:.0f}%")}
    {_kpi("Ожидание", f"{full.expectancy_r:+.2f}R", "на сделку", "win" if full.expectancy_r > 0 else "loss")}
    {_kpi("Profit factor", pf, f"у базовой линии {base_pf}")}
    {_kpi("Прибыльных окон", f"{wins} / {len(folds)}", "test R > 0")}
  </div>

  <div class="card">
    <h2>Склейка out-of-sample</h2>
    {_svg_curve(equity, base_equity)}
    <p style="color:#8a95a3;font-size:12px;margin:10px 0 0">
      Сплошная линия — walk-forward с подбором параметров.
      Пунктир — те же окна, но с параметрами по умолчанию.
    </p>
  </div>

  <div class="card">
    <h2>Окна</h2>
    {fold_table}
  </div>

  <div class="card">
    <h2>Сравнение с базовой линией</h2>
    <div class="grid">
      {_kpi("Walk-forward, R", f"{full.expectancy_r:+.2f}R", "с подбором параметров", "win" if full.expectancy_r > 0 else "loss")}
      {_kpi("По умолчанию, R", f"{base.expectancy_r:+.2f}R", "те же окна, без подбора", "win" if base.expectancy_r > 0 else "loss")}
      {_kpi("Walk-forward, итог", f"{full.total_return:+.1f}%", "", "win" if full.total_return >= 0 else "loss")}
      {_kpi("По умолчанию, итог", f"{base.total_return:+.1f}%", "", "win" if base.total_return >= 0 else "loss")}
    </div>
    <p style="color:#8a95a3;font-size:12.5px;margin:14px 0 0">
      Если подбор параметров не бьёт параметры по умолчанию — оптимизация
      отнимает время и добавляет риск переобучения, ничего не давая взамен.
    </p>
  </div>

  <div class="foot">
    Окна не пересекаются: конец проверочного отрезка одного окна — начало
    следующего. Капитал переносится из окна в окно, поэтому итог — это одна
    непрерывная история, а не сумма независимых тестов.
    <br>Не инвестиционная рекомендация.
  </div>
</div></body></html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)

