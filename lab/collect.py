"""Загрузчик исторических данных с публичных API Binance.

Ключи не нужны: используются только открытые эндпоинты свечей и funding.
Загрузка инкрементальная — повторный запуск докачивает недостающее.

Примеры:
    python -m lab.collect --symbol BTCUSDT --tf 4h --start 2019-01-01
    python -m lab.collect --symbol BTCUSDT --tf 1h --market futures --funding
    python -m lab.collect --status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from . import store

SPOT = "https://api.binance.com"
FUTURES = "https://fapi.binance.com"

TF_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

UA = {"User-Agent": "Mozilla/5.0 dsh-lab/0.1"}


class ApiError(RuntimeError):
    pass


def _get(url: str, tries: int = 5) -> list | dict:
    last = None
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code == 429 or e.code >= 500:
                wait = 2 ** attempt
                print(f"    HTTP {e.code}, пауза {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            body = e.read()[:200].decode("utf-8", "replace")
            raise ApiError(f"HTTP {e.code} на {url}: {body}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            wait = 2 ** attempt
            print(f"    сеть: {e}, пауза {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise ApiError(f"не удалось загрузить {url}: {last}")


def fetch_klines(symbol: str, market: str, tf: str, start_ms: int, end_ms: int):
    """Генератор свечей (по 1000 за запрос), без незакрытой последней свечи."""
    if tf not in TF_MS:
        raise ValueError(f"неизвестный таймфрейм {tf}; доступны: {', '.join(TF_MS)}")

    base = SPOT if market == "spot" else FUTURES
    path = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
    step = TF_MS[tf]
    now_ms = int(time.time() * 1000)
    cursor = start_ms

    while cursor <= end_ms:
        url = (
            f"{base}{path}?symbol={symbol}&interval={tf}"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        batch = _get(url)
        if not batch:
            break

        for k in batch:
            open_ms = int(k[0])
            close_ms = int(k[6])
            if close_ms > now_ms:  # свеча ещё не закрыта — не сохраняем
                continue
            yield (
                symbol,
                market,
                tf,
                open_ms,
                float(k[1]),
                float(k[2]),
                float(k[3]),
                float(k[4]),
                float(k[5]),
            )

        cursor = int(batch[-1][0]) + step
        time.sleep(0.25)


def fetch_funding(symbol: str, start_ms: int, end_ms: int):
    """История ставок финансирования (только фьючерсы, раз в 8 часов)."""
    cursor = start_ms
    while cursor <= end_ms:
        url = (
            f"{FUTURES}/fapi/v1/fundingRate?symbol={symbol}"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        batch = _get(url)
        if not batch:
            break
        for f in batch:
            yield (symbol, int(f["fundingTime"]), float(f["fundingRate"]))
        cursor = int(batch[-1]["fundingTime"]) + 1
        time.sleep(0.25)


def parse_time(value: str) -> int:
    """'2019-01-01' или '2019-01-01 12:00' или миллисекунды → ms UTC."""
    value = value.strip()
    if value.isdigit():
        return int(value)
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M"):
        try:
            dt = datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"не понимаю дату: {value}")


def _pull(con, symbol: str, market: str, tf: str, start_ms: int, end_ms: int,
          label: str) -> int:
    """Скачать диапазон и сохранить пачками."""
    total = 0
    rows: list = []
    for row in fetch_klines(symbol, market, tf, start_ms, end_ms):
        rows.append(row)
        if len(rows) >= 2000:
            total += store.upsert_candles(con, rows)
            rows = []
            print(f"  {label}: {total} свечей, последняя "
                  f"{datetime.fromtimestamp(row[3] / 1000, tz=timezone.utc):%Y-%m-%d %H:%M}")
    if rows:
        total += store.upsert_candles(con, rows)
    return total


def download(con, symbol: str, market: str, tf: str, start_ms: int, end_ms: int,
             incremental: bool, with_funding: bool) -> None:
    label = f"{symbol} {market} {tf}"
    step = TF_MS[tf]

    if incremental:
        seen = store.last_ts(con, symbol, market, tf)
        first = store.first_ts(con, symbol, market, tf)
        if seen:
            # просили более раннюю дату, чем уже есть, — добираем начало
            if first and start_ms < first:
                n = _pull(con, symbol, market, tf, start_ms, first - step,
                          label + " (начало истории)")
                if n:
                    print(f"{label}: добавлено {n} свечей в начало")
            start_ms = seen + step

    if start_ms > end_ms:
        print(f"{label}: уже актуально")
        return

    started = time.time()
    total = _pull(con, symbol, market, tf, start_ms, end_ms, label)
    took = time.time() - started
    print(f"{label}: сохранено {total} свечей за {took:.1f}s")

    if with_funding and market != "spot":
        seen = store.last_funding_ts(con, symbol) if incremental else None
        f_start = (seen + 1) if seen else start_ms
        n = 0
        buf: list = []
        for row in fetch_funding(symbol, f_start, end_ms):
            buf.append(row)
            if len(buf) >= 2000:
                n += store.upsert_funding(con, buf)
                buf = []
        if buf:
            n += store.upsert_funding(con, buf)
        print(f"{symbol} funding: {n} записей")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Загрузка свечей Binance в SQLite")
    p.add_argument("--symbol", nargs="+", default=["BTCUSDT"])
    p.add_argument("--tf", default="4h", choices=sorted(TF_MS))
    p.add_argument("--market", default="futures", choices=["spot", "futures"])
    p.add_argument("--start", type=parse_time, default=parse_time("2019-01-01"))
    p.add_argument("--end", type=parse_time, default=int(time.time() * 1000))
    p.add_argument("--funding", action="store_true", help="качать ставки финансирования")
    p.add_argument("--full", action="store_true", help="качать заново, не инкрементально")
    p.add_argument("--db", default=None)
    p.add_argument("--status", action="store_true", help="показать, что уже скачано")
    args = p.parse_args(argv)

    con = store.connect(args.db)

    if args.status:
        df = store.coverage(con)
        if df.empty:
            print("База пуста.")
        else:
            print(df.to_string(index=False))
        return 0

    for symbol in args.symbol:
        try:
            download(con, symbol.upper(), args.market, args.tf,
                     args.start, args.end, not args.full, args.funding)
        except ApiError as e:
            print(f"Ошибка на {symbol}: {e}", file=sys.stderr)
            return 1

    df = store.coverage(con)
    print("\nИтого в базе:")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
