"""Самопроверка движка на синтетических данных.

Это не тест «стратегия прибыльна», а тест «движок не врёт»:
исполнение на следующем баре, честный стоп, гэп через стоп, издержки, funding.

Запуск:  python -m lab.selftest
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from .engine import Costs, Settings, run_backtest


def bars(prices, highs=None, lows=None):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="4h", tz="UTC")
    close = np.array(prices, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": np.array(highs, dtype=float) if highs is not None else close,
            "low": np.array(lows, dtype=float) if lows is not None else close,
            "close": close,
            "volume": np.full(len(close), 1.0),
        },
        index=idx,
    )


def sig(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="4h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def dist(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="4h", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def check(name: str, ok: bool, detail: str = "") -> bool:
    mark = "OK  " if ok else "FAIL"
    print(f"{mark} {name}" + (f"  ({detail})" if detail else ""))
    return ok


def main() -> int:
    free = Costs(fee_bps=0, slippage_bps=0)
    passed = True

    # 1. Исполнение на следующем баре, а не на сигнальном.
    #    Цена 100, сигнал на баре 1, вход по открытию бара 2, выход по 120.
    df = bars([100, 100, 100, 100, 120])
    s = sig([0, 1, 1, 0, 0])
    d = dist([0, 10, 10, 0, 0])
    r = run_backtest(df, s, d, free, Settings(initial_equity=1000, risk_pct=1, max_leverage=10))
    pnl = r.trades[0].pnl if r.trades else float("nan")
    passed &= check(
        "сигнал исполняется на следующем баре",
        abs(pnl - 20.0) < 1e-6,
        f"pnl={pnl:.4f}, ожидали 20 (qty=1, движение 20)",
    )
    passed &= check(
        "вход по цене бара-исполнения, не сигнального",
        abs(r.trades[0].entry_price - 100.0) < 1e-9 and r.trades[0].entry_time == df.index[2],
        f"вход {r.trades[0].entry_time}",
    )

    # 2. Стоп внутри бара: риск ровно 1R.
    df = bars([100, 100, 100, 100, 100], highs=[100, 100, 101, 100, 100],
              lows=[100, 100, 90, 100, 100])
    s = sig([0, 1, 1, 1, 0])
    d = dist([0, 5, 5, 5, 0])
    r = run_backtest(df, s, d, free, Settings(initial_equity=1000, risk_pct=1, max_leverage=10))
    t = r.trades[0]
    passed &= check(
        "стоп даёт ровно -1R",
        abs(t.r_multiple + 1.0) < 1e-6,
        f"R={t.r_multiple:.4f}, причина={t.reason}",
    )

    # 3. Гэп через стоп: убыток больше 1R — так и должно быть.
    df = bars([100, 100, 100, 85, 85], highs=[100, 100, 101, 86, 86],
              lows=[100, 100, 99, 84, 84])
    s = sig([0, 1, 1, 1, 0])
    d = dist([0, 5, 5, 5, 0])
    r = run_backtest(df, s, d, free, Settings(initial_equity=1000, risk_pct=1, max_leverage=10))
    t = r.trades[0]
    passed &= check(
        "гэп через стоп не маскируется",
        t.r_multiple < -1.5,
        f"R={t.r_multiple:.2f} (вход 100, открытие 85, стоп 95)",
    )

    # 4. Комиссии списываются с обеих сторон.
    df = bars([100, 100, 100, 100, 100])
    s = sig([0, 1, 1, 0, 0])
    d = dist([0, 10, 10, 0, 0])
    cost = Costs(fee_bps=10, slippage_bps=0)  # 0.1% за сторону
    r = run_backtest(df, s, d, cost, Settings(initial_equity=1000, risk_pct=1, max_leverage=10))
    t = r.trades[0]
    expected = -(100 * 1 * 0.001) * 2  # оборот 100, две стороны
    passed &= check(
        "комиссия берётся дважды",
        abs(t.pnl - expected) < 1e-6,
        f"pnl={t.pnl:.4f}, ожидали {expected:.4f}",
    )

    # 5. Funding списывается с лонга при положительной ставке.
    df = bars([100] * 6)
    s = sig([1, 1, 1, 1, 1, 1])
    d = dist([10] * 6)
    idx = pd.date_range("2024-01-01", periods=4, freq="4h", tz="UTC")
    funding = pd.DataFrame({"rate": [0.01, 0.01, 0.01, 0.01]}, index=idx)
    r = run_backtest(df, s, d, free,
                     Settings(initial_equity=1000, risk_pct=1, max_leverage=10), funding)
    passed &= check(
        "funding уменьшает капитал лонга",
        r.equity.iloc[-1] < 1000,
        f"итог={r.equity.iloc[-1]:.2f} при ставке 1% за бар",
    )

    # 6. Короткая позиция не открывается без разрешения.
    df = bars([100, 100, 100, 90, 90])
    s = sig([0, -1, -1, -1, -1])
    d = dist([0, 5, 5, 5, 5])
    r = run_backtest(df, s, d, free,
                     Settings(initial_equity=1000, risk_pct=1, max_leverage=10, allow_short=False))
    passed &= check("шорт запрещён без флага", len(r.trades) == 0, f"сделок {len(r.trades)}")

    # 7. Движок не уходит в минус бесконечно: капитал ограничен снизу нулём.
    df = bars([100, 100, 100, 1, 1], highs=[100] * 5, lows=[100, 100, 100, 1, 1])
    s = sig([0, 1, 1, 1, 1])
    d = dist([0, 1, 1, 1, 1])
    r = run_backtest(df, s, d, free,
                     Settings(initial_equity=1000, risk_pct=100, max_leverage=1))
    passed &= check("капитал не отрицательный", float(r.equity.min()) >= 0,
                    f"мин. капитал {r.equity.min():.2f}")

    print()
    print("ИТОГ:", "все проверки пройдены" if passed else "ЕСТЬ ОШИБКИ")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
