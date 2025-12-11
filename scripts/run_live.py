# scripts/run_live.py

import os
import sys
import time
import hmac
import json
import base64
import hashlib
from datetime import datetime
from typing import Dict, Any, List

import httpx
import pandas as pd
import app.ta.environment

# ----------------------------------------------------------------------
# Imports GREEN V3
# ----------------------------------------------------------------------

from app.green.core import GreenV3Core, GreenStages
from app.green.styles import get_style
from app.green.pipeline.impulse import DefaultImpulseStrategy
from app.green.pipeline.pullback import DefaultPullbackStrategy
from app.green.pipeline.trigger_ai import DefaultTriggerAIStrategy
from app.green.pipeline.position import DefaultPositionStrategy
from app.exchange.okx import OkxExchange


# ----------------------------------------------------------------------
# Cliente HTTP mínimo de OKX para velas (histórico para el core)
# ----------------------------------------------------------------------

class OkxClient:
    def __init__(self):
        self.base_url = os.getenv("OKX_BASE_URL", "https://www.okx.com")
        self.api_key = os.getenv("OKX_API_KEY")
        self.api_secret = os.getenv("OKX_API_SECRET")
        self.passphrase = os.getenv("OKX_PASSPHRASE")

        if not (self.api_key and self.api_secret and self.passphrase):
            raise RuntimeError("Faltan credenciales OKX en variables de entorno.")

        self.client = httpx.Client(base_url=self.base_url, timeout=10.0)

    def _sign(self, ts: str, method: str, path: str, body: str = "") -> str:
        msg = f"{ts}{method}{path}{body}"
        mac = hmac.new(self.api_secret.encode(), msg.encode(), hashlib.sha256)
        return base64.b64encode(mac.digest()).decode()

    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        sign = self._sign(ts, method, path, body)
        headers = {
            "OK-ACCESS-KEY": self.api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": ts,
            "OK-ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
        }
        if os.getenv("OKX_ISPAPER", "0") == "1":
            headers["x-simulated-trading"] = "1"
        return headers

    def fetch_candles(self, inst_id: str, bar: str, limit: int = 300) -> pd.DataFrame:
        path = "/api/v5/market/candles"
        params = {"instId": inst_id, "bar": bar, "limit": str(limit)}

        r = self.client.get(path, params=params, headers=self._headers("GET", path))
        r.raise_for_status()
        data = r.json()

        if data.get("code") != "0":
            raise RuntimeError(f"Error OKX fetch_candles: {data}")

        rows = data.get("data", [])
        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rows = list(reversed(rows))

        ts = [int(row[0]) for row in rows]
        o = [float(row[1]) for row in rows]
        h = [float(row[2]) for row in rows]
        l = [float(row[3]) for row in rows]
        c = [float(row[4]) for row in rows]
        v = [float(row[5]) for row in rows]

        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(ts, unit="ms"),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )
        return df

OKX_INST_IDS = {
    "BTC/USDT": "BTC-USDT-SWAP",
    "ETH/USDT": "ETH-USDT-SWAP",
    "BNB/USDT": "BNB-USDT-SWAP",
    "SOL/USDT": "SOL-USDT-SWAP",
}

TF_TO_OKX_BAR = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
}


def log(msg: str):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts} UTC] {msg}", flush=True)


def build_dfs_for_symbol(okx: OkxClient, inst_id: str, style) -> Dict[str, pd.DataFrame]:
    dfs: Dict[str, pd.DataFrame] = {}

    needed_tfs = {
        style.impulse_tf,
        style.pullback_tf,
        getattr(style, "trigger_tf", style.pullback_tf),
        getattr(style, "entry_tf", style.pullback_tf),
        style.position_price_tf,
        style.position_ema_tf,
    }

    for tf in needed_tfs:
        okx_bar = TF_TO_OKX_BAR.get(tf)
        if not okx_bar:
            log(f"[WARN] No tengo mapeo OKX para TF '{tf}', lo salto.")
            continue

        try:
            df = okx.fetch_candles(inst_id, okx_bar, limit=300)
        except Exception as e:
            log(f"[{inst_id}] Error bajando velas {tf} ({okx_bar}): {e}")
            df = pd.DataFrame()

        dfs[tf] = df

    return dfs


def _append_live_trade_log(symbol: str, style_name: str, trade_row: Dict[str, Any]):
    """
    Log simple a CSV de los trades detectados por el core (entry/SL/TP/RR planificado).
    No depende de cómo termine la posición, es solo lo que se decidió abrir.
    """
    out_dir = os.getenv("GREEN_LIVE_LOG_DIR", "live_logs")
    os.makedirs(out_dir, exist_ok=True)
    sym_clean = symbol.replace("/", "")

    path = os.path.join(out_dir, f"live_trades_{sym_clean}_{style_name}.csv")

    df = pd.DataFrame([trade_row])

    if not os.path.exists(path):
        df.to_csv(path, index=False)
    else:
        df.to_csv(path, index=False, mode="a", header=False)


def main(ai_supervisor: bool = False):
    # ai_supervisor lo reutilizamos como "usar IA para decidir entry"
    style_name = os.getenv("GREEN_STYLE", "DAY").upper()
    style = get_style(style_name)

    env_syms = os.getenv("GREEN_SYMBOLS")
    if env_syms:
        symbols = [s.strip() for s in env_syms.split(",") if s.strip()]
    else:
        symbols = ["BTC/USDT"]

    cfg = style

    poll_seconds = int(os.getenv("GREEN_POLL_SECONDS", "300"))

    log(f"GREEN V3 LIVE – style={style.name}, poll={poll_seconds}s")
    log(f"Símbolos configurados: {symbols}")
    log(f"TriggerAI {'ACTIVADO' if ai_supervisor else 'DESACTIVADO'} (use_ai)")

    okx_client = OkxClient()
    exchange = OkxExchange()

    impulse_strategy = DefaultImpulseStrategy()
    pullback_strategy = DefaultPullbackStrategy()
    trigger_ai_engine = DefaultTriggerAIStrategy(use_ai=ai_supervisor)
    position_strategy = DefaultPositionStrategy(exchange=exchange)

    stages = GreenStages(
        impulse=impulse_strategy,
        pullback=pullback_strategy,
        trigger_ai=trigger_ai_engine,
        position=position_strategy,
    )

    core = GreenV3Core(
        style=style,
        stages=stages,
    )

    last_entry_seen: Dict[str, pd.Timestamp] = {}

    while True:
        for sym in symbols:
            inst_id = OKX_INST_IDS.get(sym)
            if not inst_id:
                log(f"[{sym}] No tengo instId OKX mapeado, lo salto.")
                continue

            log(f"[{sym}] Escaneando setups GREEN V3 ({style.name})...")

            dfs = build_dfs_for_symbol(okx_client, inst_id, style)

            try:
                trades = core.run(
                    symbol=sym,
                    dfs=dfs,
                    cfg=cfg
                )
            except Exception as e:
                log(f"[{sym}] Error en GreenV3Core.run(): {e}")
                continue

            if not trades:
                log(f"[{sym}] Sin trades/entradas en este escaneo.")
                continue

            df_tr = pd.DataFrame(
                [
                    {
                        "entry_time": t.entry_time,
                        "direction": t.direction,
                        "entry_price": t.entry_price,
                        "sl_initial": t.sl_initial,
                        "tp_initial": t.tp_initial,
                        "result": t.result,
                        "rr_planned": t.rr_planned,
                        "rr_real": t.rr_real,
                        "impulse_start": t.impulse_start,
                        "trigger_time": t.trigger_time,
                    }
                    for t in trades
                ]
            ).sort_values("entry_time")

            last_trade = df_tr.iloc[-1]
            last_entry_time = last_trade["entry_time"]

            prev = last_entry_seen.get(sym)
            if prev is not None and last_entry_time <= prev:
                log(f"[{sym}] No hay setups nuevos (último entry_time={last_entry_time}).")
                continue

            last_entry_seen[sym] = last_entry_time

            log(
                f"[{sym}] NUEVO TRADE detectado por GREEN V3 "
                f"({style.name}) | dir={last_trade['direction']} "
                f"entry={last_trade['entry_price']:.2f} "
                f"SL={last_trade['sl_initial']:.2f} "
                f"TP={last_trade['tp_initial']:.2f} "
                f"rr_plan={last_trade['rr_planned']:.2f}"
            )

            # Log a CSV del trade detectado
            _append_live_trade_log(
                symbol=sym,
                style_name=style.name,
                trade_row={
                    "timestamp_detected": datetime.utcnow().isoformat(),
                    **last_trade.to_dict(),
                },
            )

            # La gestión real (órdenes) ya vive en PositionStrategy + OkxExchange.
            # Si cfg.position_size > 0, DefaultPositionStrategy crea la posición real.

        log(f"Sleeping {poll_seconds} segundos...\n")
        time.sleep(poll_seconds)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GREEN V3 LIVE")
    parser.add_argument(
        "--ai",
        action="store_true",
        help="Usar IA (TriggerAI) para decidir si operar el pullback y calcular entry/SL/TP.",
    )
    args = parser.parse_args()

    main(ai_supervisor=args.ai)