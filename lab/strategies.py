"""Торговые правила.

Каждая стратегия возвращает два ряда, выровненных по барам:

    signal    — желаемая позиция по закрытию бара: +1 лонг, -1 шорт, 0 вне рынка
    stop_dist — расстояние до стопа в цене (абсолютное), известное на этом же баре

Важно: ничего не считается «наперёд». Индикаторы используют только прошлые бары,
а исполнение сигнала делает движок на следующем баре.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Signals:
    signal: pd.Series
    stop_dist: pd.Series


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range по Уайлдеру."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 60,
              atr_period: int = 14, stop_atr: float = 2.0,
              allow_short: bool = False) -> Signals:
    """Пересечение скользящих средних.

    Лонг, пока быстрая выше медленной; выход при обратном пересечении.
    Стоп — фиксированное расстояние в ATR от входа.
    """
    fast_ma = df["close"].rolling(fast, min_periods=fast).mean()
    slow_ma = df["close"].rolling(slow, min_periods=slow).mean()

    raw = pd.Series(0, index=df.index, dtype=int)
    raw[fast_ma > slow_ma] = 1
    if allow_short:
        raw[fast_ma < slow_ma] = -1
    raw[fast_ma.isna() | slow_ma.isna()] = 0

    dist = atr(df, atr_period) * stop_atr
    return Signals(signal=raw, stop_dist=dist.fillna(0.0))


def donchian(df: pd.DataFrame, entry_n: int = 55, exit_n: int = 20,
             atr_period: int = 14, stop_atr: float = 2.0,
             allow_short: bool = False) -> Signals:
    """Пробой канала Дончиана (черепаший стиль).

    Вход при закрытии выше максимума N баров (для шорта — ниже минимума),
    выход при пробое канала M баров в обратную сторону.
    """
    hh = df["high"].rolling(entry_n, min_periods=entry_n).max()
    ll = df["low"].rolling(entry_n, min_periods=entry_n).min()
    exit_hh = df["high"].rolling(exit_n, min_periods=exit_n).max()
    exit_ll = df["low"].rolling(exit_n, min_periods=exit_n).min()

    close = df["close"]
    # уровни берём со сдвигом: сегодняшний бар не должен «подсказывать» пробой
    upper = hh.shift(1)
    lower = ll.shift(1)
    exit_up = exit_hh.shift(1)
    exit_dn = exit_ll.shift(1)

    signal = pd.Series(np.nan, index=df.index, dtype=float)
    state = 0
    for i in range(len(df)):
        if np.isnan(upper.iloc[i]) or np.isnan(lower.iloc[i]):
            signal.iloc[i] = 0
            continue
        price = close.iloc[i]
        if state == 0:
            if price > upper.iloc[i]:
                state = 1
            elif allow_short and price < lower.iloc[i]:
                state = -1
        elif state == 1:
            if price < exit_dn.iloc[i]:
                state = 0
                if allow_short and price < lower.iloc[i]:
                    state = -1
        elif state == -1:
            if price > exit_up.iloc[i]:
                state = 0
                if price > upper.iloc[i]:
                    state = 1
        signal.iloc[i] = state

    stop = atr(df, atr_period) * stop_atr
    return Signals(signal=signal.astype(int), stop_dist=stop.fillna(0.0))


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False,
                                   min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, adjust=False,
                                      min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def rsi_meanrev(df: pd.DataFrame, period: int = 14, low: float = 30.0,
                high: float = 55.0, atr_period: int = 14,
                stop_atr: float = 2.0, allow_short: bool = False) -> Signals:
    """Возврат к среднему по RSI.

    Лонг при RSI < low, выход при RSI > high. Против тренда — поэтому стоп
    обязателен и обычно теснее, чем в трендовых системах.
    """
    r = rsi(df, period)
    signal = pd.Series(0, index=df.index, dtype=int)
    state = 0
    for i in range(len(df)):
        if state == 0:
            if r.iloc[i] < low:
                state = 1
            elif allow_short and r.iloc[i] > (100 - low):
                state = -1
        elif state == 1 and r.iloc[i] > high:
            state = 0
        elif state == -1 and r.iloc[i] < (100 - high):
            state = 0
        signal.iloc[i] = state

    stop = atr(df, atr_period) * stop_atr
    return Signals(signal=signal, stop_dist=stop.fillna(0.0))


REGISTRY = {
    "sma": sma_cross,
    "donchian": donchian,
    "rsi": rsi_meanrev,
}

PARAMS = {
    "sma": {"fast": 20, "slow": 60, "atr_period": 14, "stop_atr": 2.0},
    "donchian": {"entry_n": 55, "exit_n": 20, "atr_period": 14, "stop_atr": 2.0},
    "rsi": {"period": 14, "low": 30.0, "high": 55.0, "atr_period": 14, "stop_atr": 2.0},
}


def build(name: str, df: pd.DataFrame, allow_short: bool = False, **params) -> Signals:
    if name not in REGISTRY:
        raise ValueError(f"неизвестная стратегия: {name}")
    merged = dict(PARAMS[name])
    merged.update({k: v for k, v in params.items() if k in merged})
    return REGISTRY[name](df, allow_short=allow_short, **merged)
