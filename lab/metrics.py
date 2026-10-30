"""Метрики качества стратегии.

Считаем в двух валютах: в деньгах и в R (единицах риска). R важнее: он не
зависит от размера депозита и показывает качество решений.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .engine import Trade


@dataclass
class Metrics:
    bars: int = 0
    initial: float = 0.0
    final: float = 0.0
    total_return: float = 0.0
    cagr: float = 0.0
    max_dd: float = 0.0
    max_dd_pct: float = 0.0
    sharpe: float = 0.0
    exposure: float = 0.0
    trades: int = 0
    wins: int = 0
    losses: int = 0
    winrate: float = 0.0
    avg_r: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best: float = 0.0
    worst: float = 0.0
    max_loss_streak: int = 0
    avg_bars: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    buy_hold: float = 0.0

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _max_drawdown(equity: pd.Series) -> tuple[float, float]:
    peak = equity.cummax()
    dd = peak - equity
    dd_pct = (dd / peak.replace(0, np.nan)).fillna(0.0)
    return float(dd.max()), float(dd_pct.max() * 100)


def compute(
    equity: pd.Series,
    trades: list[Trade],
    periods_per_year: float,
    initial: float,
    buy_hold_equity: pd.Series | None = None,
) -> Metrics:
    m = Metrics()
    if len(equity) == 0:
        return m

    eq = equity.dropna()
    m.bars = len(eq)
    m.initial = float(initial)
    m.final = float(eq.iloc[-1])
    m.total_return = (m.final / m.initial - 1.0) * 100 if m.initial else 0.0

    years = m.bars / periods_per_year if periods_per_year else 0.0
    if years > 0 and m.initial > 0 and m.final > 0:
        m.cagr = ((m.final / m.initial) ** (1.0 / years) - 1.0) * 100
    else:
        m.cagr = -100.0

    m.max_dd, m.max_dd_pct = _max_drawdown(eq)

    rets = eq.pct_change().dropna()
    if len(rets) > 2 and rets.std() > 0:
        m.sharpe = float(rets.mean() / rets.std() * math.sqrt(periods_per_year))

    pnls = np.array([t.pnl for t in trades], dtype=float)
    rs = np.array([t.r_multiple for t in trades], dtype=float)

    m.trades = len(trades)
    if m.trades:
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        m.wins, m.losses = len(wins), len(losses)
        m.winrate = m.wins / m.trades * 100
        m.avg_r = float(rs.mean())
        m.expectancy_r = float(rs.mean())
        gross_win = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
        m.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (
            float("inf") if gross_win > 0 else 0.0
        )
        m.avg_win = float(wins.mean()) if len(wins) else 0.0
        m.avg_loss = float(losses.mean()) if len(losses) else 0.0
        m.best = float(pnls.max())
        m.worst = float(pnls.min())
        m.avg_bars = float(np.mean([t.bars for t in trades]))

        streak = 0
        for p in pnls:
            if p <= 0:
                streak += 1
                m.max_loss_streak = max(m.max_loss_streak, streak)
            else:
                streak = 0

        m.fees = float(sum(t.fees for t in trades))
        m.funding = float(sum(t.funding for t in trades))

    covered = sum(t.bars for t in trades)
    m.exposure = covered / m.bars * 100 if m.bars else 0.0

    if buy_hold_equity is not None and len(buy_hold_equity):
        bh = float(buy_hold_equity.dropna().iloc[-1])
        m.buy_hold = (bh / initial - 1.0) * 100 if initial else 0.0

    return m


def format_metrics(m: Metrics, title: str = "") -> str:
    """Текстовая таблица для консоли."""
    def pct(x: float) -> str:
        return f"{x:+.1f}%"

    pf = "∞" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"

    lines = []
    if title:
        lines.append(f"=== {title} ===")
    lines += [
        f"капитал        {m.initial:,.0f} → {m.final:,.0f}   ({pct(m.total_return)})",
        f"CAGR           {pct(m.cagr)}",
        f"макс. просадка {m.max_dd:,.0f} ({m.max_dd_pct:.1f}%)",
        f"Sharpe         {m.sharpe:.2f}",
        f"в позиции      {m.exposure:.0f}% времени",
        "",
        f"сделок         {m.trades}  (побед {m.wins}, убытков {m.losses})",
        f"winrate        {m.winrate:.1f}%",
        f"ожидание       {m.expectancy_r:+.3f}R за сделку",
        f"profit factor  {pf}",
        f"средняя победа {m.avg_win:+,.2f}   средний убыток {m.avg_loss:+,.2f}",
        f"серия убытков  {m.max_loss_streak} подряд",
        f"издержки       комиссии {m.fees:,.0f}, funding {m.funding:+,.0f}",
    ]
    if m.buy_hold:
        lines.append(f"buy & hold     {pct(m.buy_hold)} (для сравнения)")
    return "\n".join(lines)
