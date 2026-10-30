"""Локальный дашборд пейпер-трейдинга.

Поднимает мини-сервер на стандартной библиотеке: страница + JSON API.
В фоне раз в N секунд сам вызывает `paper step`, поэтому таблицы обновляются
без ручных запусков.

    python -m lab.serve --port 8765 --auto-step 300

Открыть: http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import paper

LOCK = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _read_trades() -> list[dict]:
    if not os.path.exists(paper.TRADES_PATH):
        return []
    with open(paper.TRADES_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _tail_alerts(n: int = 12) -> list[str]:
    if not os.path.exists(paper.ALERTS_PATH):
        return []
    with open(paper.ALERTS_PATH, encoding="utf-8") as f:
        return [line.rstrip() for line in f.readlines()[-n:]]


def snapshot() -> dict:
    cfg = paper._load(paper.CONFIG_PATH, {})
    state = paper._load(paper.STATE_PATH, {})
    positions = state.get("positions", {})
    last_price = state.get("last_price", {})

    capital = float(cfg.get("capital", 0) or 0)
    cash = float(state.get("cash", capital))
    equity = cash
    pos_rows = []
    for sym, p in positions.items():
        price = float(last_price.get(sym, p["entry"]))
        pnl = (price - p["entry"]) * p["qty"] * p["side"]
        equity += pnl
        pos_rows.append({
            "symbol": sym,
            "side": "лонг" if p["side"] > 0 else "шорт",
            "entry": p["entry"],
            "stop": p["stop"],
            "price": price,
            "qty": p["qty"],
            "pnl": pnl,
            "pnl_pct": (price / p["entry"] - 1) * 100 * p["side"],
            "opened": str(p.get("opened", ""))[:16].replace("T", " "),
        })

    trades = _read_trades()
    closed = len(trades)
    pnl_sum = sum(float(t["pnl"]) for t in trades) if trades else 0.0
    wins = sum(1 for t in trades if float(t["pnl"]) > 0)
    rs = [float(t["r"]) for t in trades if t.get("r") not in ("", None)]
    expect = sum(rs) / len(rs) if rs else 0.0

    # когда закроется следующий бар
    tf = cfg.get("tf", "4h")
    nxt = None
    if state.get("cursor"):
        step = {"1h": 3600, "4h": 14400, "1d": 86400}.get(tf, 14400)
        nxt = datetime.fromtimestamp(state["cursor"] / 1000 + 2 * step, tz=timezone.utc)
        nxt = nxt.strftime("%H:%M UTC")

    return {
        "strategy": cfg.get("strategy", "—"),
        "tf": tf,
        "market": cfg.get("market", "futures"),
        "symbols": len(cfg.get("symbols", [])),
        "capital": capital,
        "equity": equity,
        "return_pct": (equity / capital - 1) * 100 if capital else 0.0,
        "cash": cash,
        "open_count": len(positions),
        "positions": sorted(pos_rows, key=lambda r: -abs(r["pnl"])),
        "closed": closed,
        "pnl_sum": pnl_sum,
        "winrate": (wins / closed * 100) if closed else 0.0,
        "expectancy_r": expect,
        "updated": state.get("updated") or "—",
        "cursor": state.get("cursor"),
        "next_bar": nxt,
        "trades": trades[-12:][::-1],
        "alerts": _tail_alerts(12),
        "server_time": _now(),
    }


def run_step() -> str:
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            paper.cmd_step(None)  # type: ignore[arg-type]
    except Exception as e:  # не роняем сервер из-за сети
        return f"ошибка: {type(e).__name__}: {e}"
    return buf.getvalue().strip().splitlines()[-1] if buf.getvalue().strip() else "ok"


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Пейпер-трейдинг — живьём</title>
<style>
:root{--bg:#0e1115;--panel:#151a21;--line:#242d38;--text:#dbe2ea;--muted:#8a95a3;
--gold:#c9a227;--green:#4ea87a;--red:#cf5f5f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1120px;margin:0 auto;padding:22px 20px 50px}
header{display:flex;align-items:center;gap:12px;margin-bottom:6px}
.dot{width:9px;height:9px;border-radius:50%;background:var(--green);
box-shadow:0 0 0 0 rgba(78,168,122,.6);animation:pulse 2s infinite}
@keyframes pulse{70%{box-shadow:0 0 0 9px rgba(78,168,122,0)}100%{box-shadow:0 0 0 0 rgba(78,168,122,0)}}
h1{font-size:18px;margin:0;font-weight:650}
.sub{color:var(--muted);font-size:12.5px;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:11px 13px}
.kpi .k{font-size:10px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.kpi .v{font-size:19px;font-weight:650;margin-top:5px;font-variant-numeric:tabular-nums}
.kpi .d{font-size:11px;color:var(--muted);margin-top:3px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px 16px;margin-bottom:14px}
h2{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 10px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th{text-align:left;font-size:10px;text-transform:uppercase;letter-spacing:.05em;
color:var(--muted);padding:6px;border-bottom:1px solid var(--line)}
td{padding:6px;border-bottom:1px solid #1c242e}
.num{text-align:right;font-variant-numeric:tabular-nums}
.win{color:var(--green)}.loss{color:var(--red)}
.feed{font-family:ui-monospace,Consolas,monospace;font-size:11.5px;color:#b7c1cd;
max-height:220px;overflow:auto;line-height:1.7;white-space:pre-wrap}
.two{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:800px){.two{grid-template-columns:1fr}}
.foot{color:#5d6673;font-size:11.5px;margin-top:20px}
button{background:var(--gold);border:0;color:#171307;font-weight:650;
padding:8px 14px;border-radius:8px;cursor:pointer;font-family:inherit;font-size:12.5px}
</style></head><body><div class="wrap">

<header><span class="dot"></span><h1>Пейпер-трейдинг</h1>
<span style="margin-left:auto;color:#8a95a3;font-size:12px" id="stamp"></span></header>
<div class="sub" id="sub">загрузка…</div>

<div class="grid" id="kpis"></div>

<div class="two">
  <div class="card"><h2>Открытые позиции</h2><div id="pos"></div></div>
  <div class="card"><h2>Последние сделки</h2><div id="trades"></div></div>
</div>

<div class="card">
  <h2>События</h2>
  <div class="feed" id="alerts"></div>
  <div style="margin-top:12px"><button onclick="stepNow()">Обновить сейчас</button></div>
</div>

<div class="foot">Данные обновляются автоматически. Цены — закрытие последнего бара.
Это виртуальные деньги: ни одна сделка не уходит на биржу.</div>
</div>
<script>
const m = v => (v<0?'-$':'$') + Math.abs(v).toLocaleString('ru-RU',{maximumFractionDigits:0});
const p = v => (v>=0?'+':'') + v.toFixed(2) + '%';

function kpi(k,v,d,cls){return `<div class="kpi"><div class="k">${k}</div>
  <div class="v ${cls||''}">${v}</div><div class="d">${d||''}</div></div>`}

async function tick(){
  try{
    const d = await (await fetch('/api/state')).json();
    document.getElementById('stamp').textContent = d.server_time;
    document.getElementById('sub').textContent =
      `${d.strategy} · ${d.tf} · ${d.symbols} инструментов · обновлено ${d.updated}`
      + (d.next_bar ? ` · следующий бар ${d.next_bar}` : '');

    document.getElementById('kpis').innerHTML =
      kpi('Капитал', m(d.equity), m(d.capital)+' старт', d.return_pct>=0?'win':'loss') +
      kpi('Доход', p(d.return_pct), m(d.equity-d.capital), d.return_pct>=0?'win':'loss') +
      kpi('Сделок', d.closed, `открыто ${d.open_count}`) +
      kpi('Winrate', d.winrate.toFixed(0)+'%', '') +
      kpi('Ожидание', (d.expectancy_r>=0?'+':'')+d.expectancy_r.toFixed(2)+'R',
          'на сделку', d.expectancy_r>=0?'win':'loss');

    const pos = d.positions;
    document.getElementById('pos').innerHTML = pos.length ? `<table><thead><tr>
      <th>Инстр.</th><th>Сторона</th><th class="num">Вход</th><th class="num">Цена</th>
      <th class="num">Стоп</th><th class="num">PnL</th></tr></thead><tbody>` +
      pos.map(x=>`<tr><td><b>${x.symbol.replace('USDT','')}</b></td>
        <td>${x.side}</td><td class="num">${x.entry.toLocaleString('ru-RU',{maximumFractionDigits:2})}</td>
        <td class="num">${x.price.toLocaleString('ru-RU',{maximumFractionDigits:2})}</td>
        <td class="num" style="color:#8a95a3">${x.stop.toLocaleString('ru-RU',{maximumFractionDigits:2})}</td>
        <td class="num ${x.pnl>=0?'win':'loss'}">${m(x.pnl)}</td></tr>`).join('') +
      '</tbody></table>' : '<div style="color:#8a95a3;font-size:12.5px">Позиций нет — ждём сигнала.</div>';

    const tr = d.trades;
    document.getElementById('trades').innerHTML = tr.length ? `<table><thead><tr>
      <th>Закрыта</th><th>Инстр.</th><th>Сторона</th><th class="num">PnL</th>
      <th class="num">R</th><th>Причина</th></tr></thead><tbody>` +
      tr.map(x=>`<tr><td style="color:#8a95a3">${String(x.closed_at).slice(5,16)}</td>
        <td><b>${x.symbol.replace('USDT','')}</b></td><td>${x.side}</td>
        <td class="num ${x.pnl>=0?'win':'loss'}">${m(+x.pnl)}</td>
        <td class="num">${(+x.r>=0?'+':'')+(+x.r).toFixed(2)}R</td>
        <td style="color:#8a95a3">${x.reason}</td></tr>`).join('') +
      '</tbody></table>' : '<div style="color:#8a95a3;font-size:12.5px">Сделок ещё нет.</div>';

    document.getElementById('alerts').textContent =
      d.alerts.length ? d.alerts.join('\n') : 'пока тихо';
  }catch(e){ console.error(e); }
}

async function stepNow(){
  await fetch('/api/step', {method:'POST'});
  tick();
}

tick();
setInterval(tick, 10000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: str, ctype: str = "application/json; charset=utf-8"):
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/api/state"):
            self._send(200, json.dumps(snapshot(), ensure_ascii=False))
        elif self.path in ("/", "/index.html"):
            self._send(200, PAGE, "text/html; charset=utf-8")
        else:
            self._send(404, '{"error":"not found"}')

    def do_POST(self):
        if self.path.startswith("/api/step"):
            with LOCK:
                result = run_step()
            print(f"[{_now()}] step: {result}")
            self._send(200, json.dumps(snapshot(), ensure_ascii=False))
        else:
            self._send(404, '{"error":"not found"}')

    def log_message(self, fmt, *args):
        pass


def auto_loop(interval: int):
    while True:
        time.sleep(interval)
        with LOCK:
            result = run_step()
        print(f"[{_now()}] автошаг: {result}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Живой дашборд пейпер-трейдинга")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--auto-step", type=int, default=300,
                   help="интервал автошагов в секундах, 0 — выключить")
    args = p.parse_args(argv)

    if not os.path.exists(paper.CONFIG_PATH):
        print("Нет конфига. Сначала: python -m lab.paper init", flush=True)
        return 1

    # первый шаг сразу, чтобы поймать точку отсчёта
    with LOCK:
        print(f"[{_now()}] старт: {run_step()}", flush=True)

    if args.auto_step > 0:
        threading.Thread(target=auto_loop, args=(args.auto_step,), daemon=True).start()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"дашборд: http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nостановлено", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


