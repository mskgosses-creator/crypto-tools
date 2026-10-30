"""Движок бэктеста: бары, издержки, стопы, funding.

Принципы, от которых не отступаем:

* Сигнал считается по закрытию бара i и исполняется по открытию бара i+1.
  Иначе появляется look-ahead и бэктест начинает врать.
* Комиссия берётся с каждой стороны, слippage — тоже (всегда против нас).
* Стоп проверяется внутри бара; если бар открылся уже за стопом (гэп),
  выходим по открытию, а не по цене стопа.
* Если в одном баре задеты и стоп, и тейк — считаем, что первым сработал стоп.
* Позиция закрывается по последнему бару, даже если сигнала на выход не было.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Costs:
    """Издержки в базисных пунктах (1 bp = 0.01%)."""

    fee_bps: float = 5.0        # 0.05% за сторону — типичный тейкер на фьючерсах
    slippage_bps: float = 2.0   # 0.02% проскальзывание


@dataclass
class Settings:
    initial_equity: float = 1000.0
    risk_pct: float = 1.0       # риск на сделку, % от капитала
    max_leverage: float = 1.0   # ограничение размера позиции
    allow_short: bool = False
    trade_from: int = 0         # индекс бара, с которого разрешено открываться


@dataclass
class Trade:
    side: int                   # +1 лонг, -1 шорт
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    fees: float
    funding: float
    risk: float
    reason: str
    bars: int

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.risk if self.risk > 0 else 0.0


@dataclass
class Result:
    equity: pd.Series
    trades: list[Trade]
    exposure: float

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(
                columns=[
                    "side", "entry_time", "exit_time", "entry_price", "exit_price",
                    "qty", "pnl", "fees", "funding", "risk", "r", "reason", "bars",
                ]
            )
        return pd.DataFrame(
            [
                {
                    "side": "лонг" if t.side > 0 else "шорт",
                    "entry_time": t.entry_time,
                    "exit_time": t.exit_time,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "qty": t.qty,
                    "pnl": t.pnl,
                    "fees": t.fees,
                    "funding": t.funding,
                    "risk": t.risk,
                    "r": t.r_multiple,
                    "reason": t.reason,
                    "bars": t.bars,
                }
                for t in self.trades
            ]
        )


def _funding_per_bar(df: pd.DataFrame, funding: pd.DataFrame | None) -> np.ndarray:
    """Сумма ставок финансирования, попавших в каждый бар."""
    out = np.zeros(len(df), dtype=float)
    if funding is None or len(funding) == 0:
        return out
    pos = df.index.searchsorted(funding.index, side="right") - 1
    rates = funding["rate"].to_numpy(dtype=float)
    for idx, rate in zip(pos, rates):
        if 0 <= idx < len(out):
            out[idx] += rate
    return out


def run_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    stop_dist: pd.Series,
    costs: Costs | None = None,
    settings: Settings | None = None,
    funding: pd.DataFrame | None = None,
) -> Result:
    """Прогнать стратегию по барам.

    signal:    желаемая позиция по закрытию бара (+1 / 0 / -1).
    stop_dist: расстояние до стопа в цене на момент сигнала (абсолютное).
    """
    if costs is None:
        costs = Costs()
    if settings is None:
        settings = Settings()

    n = len(df)
    if n == 0:
        return Result(pd.Series(dtype=float), [], 0.0)

    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    sig = np.nan_to_num(signal.astype(float).to_numpy(), nan=0.0)
    sd = np.nan_to_num(stop_dist.astype(float).to_numpy(), nan=0.0)
    fund = _funding_per_bar(df, funding)

    fee_rate = costs.fee_bps / 10_000.0
    slip = costs.slippage_bps / 10_000.0

    equity = float(settings.initial_equity)
    curve = np.empty(n, dtype=float)
    trades: list[Trade] = []

    side = 0
    qty = 0.0
    entry_px = 0.0
    stop_px = 0.0
    entry_i = 0
    entry_equity = 0.0
    risk_amt = 0.0
    fees_paid = 0.0
    fund_paid = 0.0
    pending: tuple[int, float] | None = None
    bars_in_position = 0

    def close_position(price: float, i: int, reason: str) -> None:
        nonlocal side, qty, equity, fees_paid, fund_paid, bars_in_position
        if side == 0:
            return
        # выход всегда по худшей для нас цене
        fill = price * (1 - slip) if side > 0 else price * (1 + slip)
        gross = (fill - entry_px) * qty * side
        fee = abs(fill * qty) * fee_rate
        equity += gross - fee
        fees_paid += fee
        trades.append(
            Trade(
                side=side,
                entry_time=df.index[entry_i],
                exit_time=df.index[i],
                entry_price=entry_px,
                exit_price=fill,
                qty=qty,
                pnl=gross - fee - (fees_paid_entry + fund_paid),
                fees=fee + fees_paid_entry,
                funding=fund_paid,
                risk=risk_amt,
                reason=reason,
                bars=max(1, i - entry_i + 1),
            )
        )
        side, qty, bars_in_position = 0, 0.0, 0
        fees_paid, fund_paid = 0.0, 0.0

    fees_paid_entry = 0.0

    for i in range(n):
        # 1. исполнение сигнала предыдущего бара по открытию текущего
        if pending is not None:
            target, dist = pending
            pending = None
            if side != 0 and target != side:
                close_position(o[i], i, "signal")
            if target != 0 and side == 0 and i >= settings.trade_from:
                dist = dist if dist > 0 else 0.0
                if dist > 0 and equity > 0:
                    fill = o[i] * (1 + slip) if target > 0 else o[i] * (1 - slip)
                    risk_take = equity * settings.risk_pct / 100.0
                    size = risk_take / dist
                    notional = size * fill
                    cap = equity * settings.max_leverage
                    if notional > cap:
                        size = cap / fill
                        notional = cap
                    if size > 0:
                        side = target
                        qty = size
                        entry_px = fill
                        entry_i = i
                        entry_equity = equity
                        risk_amt = size * dist
                        stop_px = fill - dist if side > 0 else fill + dist
                        fee = notional * fee_rate
                        equity -= fee
                        fees_paid_entry = fee
                        fees_paid = fee
                        fund_paid = 0.0
                        bars_in_position = 0

        # 2. стоп: сначала гэп через стоп на открытии
        if side != 0:
            if (side > 0 and o[i] <= stop_px) or (side < 0 and o[i] >= stop_px):
                close_position(o[i], i, "stop")
            elif (side > 0 and low[i] <= stop_px) or (side < 0 and h[i] >= stop_px):
                close_position(stop_px, i, "stop")

        # 3. финансирование
        if side != 0 and fund[i] != 0.0:
            pay = qty * c[i] * fund[i] * side
            equity -= pay
            fund_paid += pay

        # 4. переоценка
        if side != 0:
            bars_in_position += 1
            unreal = (c[i] - entry_px) * qty * side
            curve[i] = equity + unreal
        else:
            curve[i] = equity

        # 5. сигнал по закрытию бара — в очередь на следующий бар
        want = int(np.sign(sig[i]))
        if not settings.allow_short and want < 0:
            want = 0
        pending = (want, sd[i])

        if equity <= 0:
            equity = 0.0
            break

    # закрываем всё, что осталось, по последнему закрытию
    if side != 0:
        close_position(c[-1], n - 1, "end")
        curve[n - 1] = equity

    filled = np.where(curve == 0.0, np.nan, curve)
    series = pd.Series(filled, index=df.index, name="equity").ffill()
    if series.isna().all():
        series = pd.Series(np.full(n, float(settings.initial_equity)), index=df.index, name="equity")

    exposure = float(np.mean([1.0 if t.bars > 0 else 0.0 for t in trades])) if trades else 0.0
    covered = sum(t.bars for t in trades)
    exposure = covered / n if n else 0.0

    return Result(equity=series, trades=trades, exposure=exposure)


def buy_and_hold(df: pd.DataFrame, costs: Costs | None = None,
                 initial_equity: float = 1000.0) -> pd.Series:
    """Купить на первом баре и держать — базовая линия для сравнения."""
    if costs is None:
        costs = Costs()
    if len(df) == 0:
        return pd.Series(dtype=float)
    fee = costs.fee_bps / 10_000.0
    slip = costs.slippage_bps / 10_000.0

    entry = float(df["open"].iloc[0]) * (1 + slip)
    qty = (initial_equity * (1 - fee)) / entry
    return (df["close"] * qty).rename("buy_hold")
