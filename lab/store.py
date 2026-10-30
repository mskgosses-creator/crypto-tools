"""Хранилище свечей и funding в SQLite.

Одна база на всё: свечи разных инструментов, таймфреймов и рынков (spot/futures)
плюс история ставок финансирования для фьючерсов.
"""

from __future__ import annotations

import os
import sqlite3
from typing import Iterable, Sequence

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_PATH = os.path.join(ROOT, "data", "market.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS candles (
    symbol  TEXT    NOT NULL,
    market  TEXT    NOT NULL,
    tf      TEXT    NOT NULL,
    ts      INTEGER NOT NULL,
    open    REAL    NOT NULL,
    high    REAL    NOT NULL,
    low     REAL    NOT NULL,
    close   REAL    NOT NULL,
    volume  REAL    NOT NULL,
    PRIMARY KEY (symbol, market, tf, ts)
);

CREATE TABLE IF NOT EXISTS funding (
    symbol  TEXT    NOT NULL,
    ts      INTEGER NOT NULL,
    rate    REAL    NOT NULL,
    PRIMARY KEY (symbol, ts)
);

CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles (symbol, market, tf, ts);
CREATE INDEX IF NOT EXISTS idx_funding_lookup
    ON funding (symbol, ts);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    """Открыть базу, создав её при необходимости."""
    p = path or DEFAULT_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    con = sqlite3.connect(p)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def upsert_candles(con: sqlite3.Connection, rows: Iterable[Sequence]) -> int:
    """rows: (symbol, market, tf, ts, open, high, low, close, volume)."""
    cur = con.executemany(
        "INSERT OR REPLACE INTO candles "
        "(symbol, market, tf, ts, open, high, low, close, volume) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.commit()
    return cur.rowcount


def upsert_funding(con: sqlite3.Connection, rows: Iterable[Sequence]) -> int:
    """rows: (symbol, ts, rate)."""
    cur = con.executemany(
        "INSERT OR REPLACE INTO funding (symbol, ts, rate) VALUES (?, ?, ?)",
        rows,
    )
    con.commit()
    return cur.rowcount


def last_ts(con: sqlite3.Connection, symbol: str, market: str, tf: str) -> int | None:
    row = con.execute(
        "SELECT MAX(ts) FROM candles WHERE symbol=? AND market=? AND tf=?",
        (symbol, market, tf),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def first_ts(con: sqlite3.Connection, symbol: str, market: str, tf: str) -> int | None:
    row = con.execute(
        "SELECT MIN(ts) FROM candles WHERE symbol=? AND market=? AND tf=?",
        (symbol, market, tf),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def last_funding_ts(con: sqlite3.Connection, symbol: str) -> int | None:
    row = con.execute(
        "SELECT MAX(ts) FROM funding WHERE symbol=?", (symbol,)
    ).fetchone()
    return row[0] if row and row[0] is not None else None


def load_candles(
    con: sqlite3.Connection,
    symbol: str,
    market: str,
    tf: str,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    """Свечи в DataFrame с DatetimeIndex (UTC) и колонками ohlcv."""
    sql = (
        "SELECT ts, open, high, low, close, volume FROM candles "
        "WHERE symbol=? AND market=? AND tf=?"
    )
    params: list = [symbol, market, tf]
    if start is not None:
        sql += " AND ts >= ?"
        params.append(start)
    if end is not None:
        sql += " AND ts <= ?"
        params.append(end)
    sql += " ORDER BY ts"

    df = pd.read_sql_query(sql, con, params=params)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("time").drop(columns=["ts"])
    return df[["open", "high", "low", "close", "volume"]].astype("float64")


def load_funding(
    con: sqlite3.Connection, symbol: str, start: int | None = None, end: int | None = None
) -> pd.DataFrame:
    """Ставки финансирования с DatetimeIndex (UTC)."""
    sql = "SELECT ts, rate FROM funding WHERE symbol=?"
    params: list = [symbol]
    if start is not None:
        sql += " AND ts >= ?"
        params.append(start)
    if end is not None:
        sql += " AND ts <= ?"
        params.append(end)
    sql += " ORDER BY ts"

    df = pd.read_sql_query(sql, con, params=params)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("time")[["rate"]]


def coverage(con: sqlite3.Connection) -> pd.DataFrame:
    """Что уже скачано: по каждой паре symbol/market/tf — строки и период."""
    sql = """
    SELECT symbol, market, tf, COUNT(*) AS bars,
           MIN(ts) AS first_ts, MAX(ts) AS last_ts
    FROM candles
    GROUP BY symbol, market, tf
    ORDER BY symbol, market, tf
    """
    df = pd.read_sql_query(sql, con)
    if df.empty:
        return df
    df["from"] = pd.to_datetime(df["first_ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    df["to"] = pd.to_datetime(df["last_ts"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return df[["symbol", "market", "tf", "bars", "from", "to"]]
