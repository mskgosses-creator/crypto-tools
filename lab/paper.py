"""Пейпер-трейдинг: те же правила, на живых данных, без денег.

Логика намеренно скучная и повторяет бэктест:

1. Берём последние закрытые свечи с биржи и складываем в базу.
2. Для каждой новой свечи считаем сигнал стратегии (он причинный — только прошлое).
3. Открываем/закрываем виртуальные позиции по цене закрытия бара со слippage
   и комиссией, стоп — внутри бара, funding — по фактическим ставкам.
4. Состояние храним в `paper/state.json`, сделки пишем в `paper/trades.csv`,
   события — в `paper/alerts.log` (и в Telegram, если заданы переменные окружения).

Запуск вручную или по расписанию:

    python -m lab.paper init --strategy donchian --capital 10000
    python -m lab.paper step
    python -m lab.paper status
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import strategies, store
from .collect import TF_MS, fetch_last, fetch_recent_funding, parse_time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PAPER_DIR = os.path.join(ROOT, "paper")
CONFIG_PATH = os.path.join(PAPER_DIR, "config.json")
STATE_PATH = os.path.join(PAPER_DIR, "state.json")
TRADES_PATH = os.path.join(PAPER_DIR, "trades.csv")
ALERTS_PATH = os.path.join(PAPER_DIR, "alerts.log")

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT",
]


# ---------------------------------------------------------------- состояние

def _load(path: str, default: dict) -> dict:
    if not os.path.exists(path):
        return dict(default)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _ts_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 10**6)


# ------------------------------------------------------------------ алерты

def notify(lines: list[str], cfg: dict) -> None:
    """Печать + лог + опционально Telegram."""
    if not lines:
        return
    stamp = _now()
    os.makedirs(PAPER_DIR, exist_ok=True)
    with open(ALERTS_PATH, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(f"[{stamp}] {line}\n")

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return
    text = "\n".join(lines)
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
        urllib.request.urlopen(req, timeout=10).read()
    except Exception as e:  # сеть/токен — не повод падать
        print(f"  (telegram не отправлен: {str(e)[:60]})", file=sys.stderr)


# --------------------------------------------------------------------- init

def cmd_init(args) -> int:
    if os.path.exists(CONFIG_PATH) and not args.force:
        print(f"Конфиг уже есть: {CONFIG_PATH}\nПерезаписать: --force")
        return 1

    cfg = {
        "strategy": args.strategy,
        "symbols": [s.upper() for s in args.symbols],
        "tf": args.tf,
        "market": args.market,
        "capital": args.capital,
        "risk_pct": args.risk,
        "max_leverage": args.leverage,
        "fee_bps": args.fee_bps,
        "slip_bps": args.slip_bps,
        "created": _now(),
    }
    state = {
        "created": _now(),
        "updated": None,
        "cursor": None,
        "cash": float(args.capital),
        "positions": {},
        "last_price": {},
        "closed_trades": 0,
        "bars_processed": 0,
    }
    if args.from_date:
        ts = pd.Timestamp(args.from_date, unit="ms", tz="UTC")
        state["cursor"] = _ts_ms(ts)
        print(f"старт с {ts:%Y-%m-%d %H:%M} UTC — step догонит историю с этой даты")

    _save(CONFIG_PATH, cfg)
    _save(STATE_PATH, state)
    print(f"инициализировано: {len(cfg['symbols'])} инструментов, "
          f"капитал {cfg['capital']:,.0f}, стратегия {cfg['strategy']}")
    print(f"  {CONFIG_PATH}\n  {STATE_PATH}")
    return 0


# --------------------------------------------------------------------- step

def _funding_map(funding: pd.DataFrame | None) -> dict[int, float]:
    if funding is None or len(funding) == 0:
        return {}
    return {_ts_ms(ts): float(r) for ts, r in funding["rate"].items()}


def cmd_step(args) -> int:
    cfg = _load(CONFIG_PATH, {})
    if not cfg:
        print("Нет конфига. Сначала: python -m lab.paper init", file=sys.stderr)
        return 1
    state = _load(STATE_PATH, {})
    symbols = cfg["symbols"]
    tf, market = cfg["tf"], cfg["market"]

    con = store.connect()
    n = len(symbols)
    per = cfg["capital"] / n

    # 1. свежие данные
    for sym in symbols:
        rows = fetch_last(sym, market, tf, 400)
        if rows:
            store.upsert_candles(con, rows)
        if market == "futures":
            fr = fetch_recent_funding(sym, 60)
            if fr:
                store.upsert_funding(con, fr)

    # 2. сигналы на всей истории (они причинные)
    data: dict[str, tuple] = {}
    for sym in symbols:
        df = store.load_candles(con, sym, market, tf)
        if len(df) < 300:
            continue
        sig = strategies.build(cfg["strategy"], df)
        funding = store.load_funding(con, sym) if market == "futures" else None
        data[sym] = (df, sig, _funding_map(funding))

    if not data:
        print("Нет данных для торговли.", file=sys.stderr)
        return 1

    last_closed = max(df.index[-1] for df, _, _ in data.values())
    cursor = state.get("cursor")
    if cursor is None:
        state["cursor"] = _ts_ms(last_closed)
        state["updated"] = _now()
        _save(STATE_PATH, state)
        print(f"точка отсчёта: {last_closed:%Y-%m-%d %H:%M} UTC. "
              f"Ждём следующего закрытого бара.")
        return 0

    step_ms = TF_MS[tf]
    fee_rate = cfg["fee_bps"] / 10_000
    slip = cfg["slip_bps"] / 10_000

    # все новые бары по всем инструментам, по возрастанию времени
    times = sorted({
        ts for df, _, _ in data.values()
        for ts in df.index
        if _ts_ms(ts) > cursor
    })
    if not times:
        print("Новых закрытых баров нет — ждём.")
        return 0

    alerts: list[str] = []
    trades_out: list[dict] = []
    processed = 0

    def equity_now() -> float:
        eq = state["cash"]
        for s, p in state["positions"].items():
            price = state["last_price"].get(s, p["entry"])
            eq += (price - p["entry"]) * p["qty"] * p["side"]
        return eq

    for ts in times:
        for sym, (df, sig, fund) in data.items():
            if ts not in df.index:
                continue
            i = df.index.get_loc(ts)
            o = float(df["open"].iloc[i])
            high = float(df["high"].iloc[i])
            low = float(df["low"].iloc[i])
            close = float(df["close"].iloc[i])
            state["last_price"][sym] = close
            processed += 1

            pos = state["positions"].get(sym)

            # --- стоп внутри бара
            if pos:
                if (pos["side"] > 0 and low <= pos["stop"]) or \
                   (pos["side"] < 0 and high >= pos["stop"]):
                    fill = pos["stop"] * (1 - slip if pos["side"] > 0 else 1 + slip)
                    pnl = (fill - pos["entry"]) * pos["qty"] * pos["side"]
                    fee = abs(fill * pos["qty"]) * fee_rate
                    state["cash"] += pnl - fee
                    r = pnl / pos["risk"] if pos["risk"] else 0.0
                    trades_out.append({
                        "closed_at": ts, "symbol": sym,
                        "side": "лонг" if pos["side"] > 0 else "шорт",
                        "entry": pos["entry"], "exit": fill, "qty": pos["qty"],
                        "pnl": pnl - fee, "r": r, "reason": "stop",
                        "opened": pos["opened"], "bars": pos["bars"],
                    })
                    alerts.append(
                        f"СТОП {sym} {'лонг' if pos['side'] > 0 else 'шорт'} "
                        f"выход {fill:,.2f}, {r:+.2f}R ({pnl - fee:+,.0f})"
                    )
                    del state["positions"][sym]
                    state["closed_trades"] += 1
                    pos = None

            # --- финансирование
            if pos and fund:
                rate = fund.get(_ts_ms(ts), 0.0)
                if rate:
                    state["cash"] -= pos["qty"] * close * rate * pos["side"]

            # --- сигнал
            want = int(np.sign(sig.signal.iloc[i]))
            current = pos["side"] if pos else 0
            if want != current:
                if pos:
                    fill = close * (1 - slip if pos["side"] > 0 else 1 + slip)
                    pnl = (fill - pos["entry"]) * pos["qty"] * pos["side"]
                    fee = abs(fill * pos["qty"]) * fee_rate
                    state["cash"] += pnl - fee
                    r = pnl / pos["risk"] if pos["risk"] else 0.0
                    trades_out.append({
                        "closed_at": ts, "symbol": sym,
                        "side": "лонг" if pos["side"] > 0 else "шорт",
                        "entry": pos["entry"], "exit": fill, "qty": pos["qty"],
                        "pnl": pnl - fee, "r": r, "reason": "signal",
                        "opened": pos["opened"], "bars": pos["bars"],
                    })
                    alerts.append(
                        f"ВЫХОД {sym} {'лонг' if pos['side'] > 0 else 'шорт'} "
                        f"по сигналу {fill:,.2f}, {r:+.2f}R ({pnl - fee:+,.0f})"
                    )
                    del state["positions"][sym]
                    state["closed_trades"] += 1
                    pos = None

                if want != 0:
                    dist = float(sig.stop_dist.iloc[i])
                    eq = equity_now()
                    if dist > 0 and eq > 0:
                        risk_amt = (eq / n) * cfg["risk_pct"] / 100
                        qty = risk_amt / dist
                        notional = qty * close
                        cap = (eq / n) * cfg["max_leverage"]
                        if notional > cap:
                            qty = cap / close
                            notional = cap
                        if qty > 0:
                            fill = close * (1 + slip if want > 0 else 1 - slip)
                            fee = notional * fee_rate
                            state["cash"] -= fee
                            state["positions"][sym] = {
                                "side": want,
                                "qty": qty,
                                "entry": fill,
                                "stop": fill - dist * want,
                                "risk": qty * dist,
                                "opened": str(ts),
                                "bars": 0,
                                "fees": fee,
                            }
                            alerts.append(
                                f"ВХОД {sym} {'лонг' if want > 0 else 'шорт'} "
                                f"{qty:.6g} @ {fill:,.2f}, стоп {fill - dist * want:,.2f}, "
                                f"риск {qty * dist:,.0f}"
                            )
            if sym in state["positions"]:
                state["positions"][sym]["bars"] = state["positions"][sym].get("bars", 0) + 1

    state["cursor"] = _ts_ms(times[-1])
    state["updated"] = _now()
    state["bars_processed"] = state.get("bars_processed", 0) + processed
    _save(STATE_PATH, state)

    if trades_out:
        _append_trades(trades_out)

    notify(alerts, cfg)

    eq = equity_now()
    print(f"обработано баров: {processed}, событий: {len(alerts)}")
    for a in alerts:
        print("  " + a)
    print(f"капитал: {cfg['capital']:,.0f} → {eq:,.0f} "
          f"({(eq / cfg['capital'] - 1) * 100:+.2f}%), "
          f"открытых позиций: {len(state['positions'])}")
    return 0


def _append_trades(rows: list[dict]) -> None:
    os.makedirs(PAPER_DIR, exist_ok=True)
    fields = ["closed_at", "symbol", "side", "entry", "exit", "qty",
              "pnl", "r", "reason", "opened", "bars"]
    new_file = not os.path.exists(TRADES_PATH)
    with open(TRADES_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


# ------------------------------------------------------------------- status

def cmd_status(args) -> int:
    cfg = _load(CONFIG_PATH, {})
    state = _load(STATE_PATH, {})
    if not cfg:
        print("Нет конфига. Сначала: python -m lab.paper init", file=sys.stderr)
        return 1

    cash = state.get("cash", cfg["capital"])
    eq = cash
    for s, p in state.get("positions", {}).items():
        price = state.get("last_price", {}).get(s, p["entry"])
        eq += (price - p["entry"]) * p["qty"] * p["side"]

    print(f"стратегия   {cfg['strategy']} · {cfg['tf']} · {len(cfg['symbols'])} инструментов")
    print(f"капитал     {cfg['capital']:,.0f} → {eq:,.0f} ({(eq / cfg['capital'] - 1) * 100:+.2f}%)")
    print(f"обновлено   {state.get('updated') or '—'}")
    print(f"сделок      {state.get('closed_trades', 0)} закрыто")

    positions = state.get("positions", {})
    if not positions:
        print("позиций нет")
    else:
        print("\nоткрытые позиции:")
        print(f"  {'инстр.':10s} {'сторона':7s} {'вход':>11s} {'стоп':>11s} "
              f"{'кол-во':>10s} {'PnL':>10s}")
        for s, p in positions.items():
            price = state.get("last_price", {}).get(s, p["entry"])
            pnl = (price - p["entry"]) * p["qty"] * p["side"]
            print(f"  {s:10s} {'лонг' if p['side'] > 0 else 'шорт':7s} "
                  f"{p['entry']:>11,.2f} {p['stop']:>11,.2f} "
                  f"{p['qty']:>10.6g} {pnl:>+10,.0f}")

    if os.path.exists(TRADES_PATH):
        df = pd.read_csv(TRADES_PATH)
        if len(df):
            wins = (df["pnl"] > 0).sum()
            print(f"\nзакрытых сделок в журнале: {len(df)}, "
                  f"winrate {wins / len(df) * 100:.0f}%, "
                  f"сумма PnL {df['pnl'].sum():+,.0f}, "
                  f"ожидание {df['r'].mean():+.2f}R")

    if args.alerts and os.path.exists(ALERTS_PATH):
        with open(ALERTS_PATH, encoding="utf-8") as f:
            lines = f.readlines()[-15:]
        print("\nпоследние события:")
        for line in lines:
            print("  " + line.rstrip())
    return 0


# -------------------------------------------------------------------- reset

def cmd_reset(args) -> int:
    if not args.yes:
        print("Удалить состояние и журнал? Запустите с --yes")
        return 1
    for path in (STATE_PATH, TRADES_PATH, ALERTS_PATH):
        if os.path.exists(path):
            os.remove(path)
    print("состояние и журнал удалены (конфиг сохранён)")
    return 0


# ---------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Пейпер-трейдинг на живых данных")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("init", help="создать конфиг и состояние")
    pi.add_argument("--strategy", default="donchian", choices=sorted(strategies.REGISTRY))
    pi.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    pi.add_argument("--tf", default="4h", choices=sorted(TF_MS))
    pi.add_argument("--market", default="futures", choices=["spot", "futures"])
    pi.add_argument("--capital", type=float, default=10_000.0)
    pi.add_argument("--risk", type=float, default=1.0)
    pi.add_argument("--leverage", type=float, default=1.0)
    pi.add_argument("--fee-bps", type=float, default=5.0)
    pi.add_argument("--slip-bps", type=float, default=2.0)
    pi.add_argument("--from", dest="from_date", type=parse_time, default=None,
                    help="догнать историю с этой даты (по умолчанию — с текущего бара)")
    pi.add_argument("--force", action="store_true")
    pi.set_defaults(func=cmd_init)

    ps = sub.add_parser("step", help="обработать новые закрытые бары")
    ps.set_defaults(func=cmd_step)

    pt = sub.add_parser("status", help="показать состояние")
    pt.add_argument("--alerts", action="store_true", help="плюс последние события")
    pt.set_defaults(func=cmd_status)

    pr = sub.add_parser("reset", help="удалить состояние и журнал")
    pr.add_argument("--yes", action="store_true")
    pr.set_defaults(func=cmd_reset)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

