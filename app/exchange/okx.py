# app/exchange/okx.py
"""
Implementación LIVE de IExchange usando el broker OKX.

Objetivos:
    - NO contener lógica de GREEN V3.
    - Solo crear, modificar y cerrar posiciones reales en OKX.
    - Exponer velas reales con fetch_candles().
    - Mantener un registro local mínimo de posiciones.

Requiere variables de entorno:
    OKX_API_KEY
    OKX_API_SECRET
    OKX_API_PASS
    OKX_ISPAPER   ("0" live, "1" paper trading)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

import pandas as pd

from app.exchange.base import IExchange

try:
    from okx.Account_api import AccountAPI
    from okx.Market_api import MarketAPI
    from okx.Trade_api import TradeAPI
except Exception:
    raise ImportError("Falta instalar el SDK oficial de OKX: pip install okx-python")


@dataclass
class OkxPosition:
    position_id: str
    symbol: str
    side: str          # "long" o "short"
    size: float
    entry_price: float
    sl: float
    tp: float
    okx_order_id: str
    open_time: datetime
    is_open: bool = True
    close_time: Optional[datetime] = None
    close_reason: Optional[str] = None


class OkxExchange(IExchange):
    def __init__(self):
        api_key = os.getenv("OKX_API_KEY")
        api_secret = os.getenv("OKX_API_SECRET")
        api_pass = os.getenv("OKX_API_PASS")
        flag = os.getenv("OKX_ISPAPER", "1")
        
        if not api_key or not api_secret or not api_pass:
            raise RuntimeError("Faltan credenciales OKX en variables de entorno.")

        self.market = MarketAPI(api_key, api_secret, api_pass, False, flag=flag)
        self.trade = TradeAPI(api_key, api_secret, api_pass, False, flag=flag)
        self.account = AccountAPI(api_key, api_secret, api_pass, False, flag=flag)

        self._positions: Dict[str, OkxPosition] = {}

    # ----------------------------- OHLCV -----------------------------
    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        params = {
            "instId": symbol.replace("/", "-"),
            "bar": timeframe,
            "limit": 300,
        }

        if start:
            params["before"] = int(start.timestamp() * 1000)
        if end:
            params["after"] = int(end.timestamp() * 1000)

        data = self.market.get_history_candlesticks(**params)

        if "data" not in data or not data["data"]:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rows = []
        for ohlc in data["data"]:
            ts = datetime.fromtimestamp(int(ohlc[0]) / 1000)
            rows.append(
                {
                    "timestamp": ts,
                    "open": float(ohlc[1]),
                    "high": float(ohlc[2]),
                    "low": float(ohlc[3]),
                    "close": float(ohlc[4]),
                    "volume": float(ohlc[5]),
                }
            )

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

        if start:
            df = df[df["timestamp"] >= start]
        if end:
            df = df[df["timestamp"] <= end]

        return df

    # ------------------------- create_position -----------------------
    def create_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        sl: float,
        tp: float,
    ) -> str:
        inst = symbol.replace("/", "-")
        side_okx = "buy" if side == "long" else "sell"

        order = self.trade.place_order(
            instId=inst,
            tdMode="cross",
            side=side_okx,
            ordType="market",
            sz=str(size),
        )

        if "data" not in order or not order["data"]:
            raise RuntimeError(f"Error creando orden en OKX: {order}")

        okx_id = order["data"][0]["ordId"]
        now = datetime.utcnow()
        pos_id = f"{symbol}-{side}-{okx_id}"

        self._positions[pos_id] = OkxPosition(
            position_id=pos_id,
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            okx_order_id=okx_id,
            open_time=now,
        )

        # SL / TP reales (2 órdenes "reduce only")
        self._submit_sl_tp(symbol, side, size, sl, tp)

        return pos_id

    def _submit_sl_tp(self, symbol: str, side: str, size: float, sl: float, tp: float):
        inst = symbol.replace("/", "-")
        stop_side = "sell" if side == "long" else "buy"
        tp_side = "sell" if side == "long" else "buy"

        # STOP LOSS
        self.trade.place_order(
            instId=inst,
            tdMode="cross",
            side=stop_side,
            ordType="trigger",
            sz=str(size),
            triggerPx=str(sl),
            reduceOnly=True,
        )

        # TAKE PROFIT
        self.trade.place_order(
            instId=inst,
            tdMode="cross",
            side=tp_side,
            ordType="trigger",
            sz=str(size),
            triggerPx=str(tp),
            reduceOnly=True,
        )

    # ------------------------- update_sl_tp --------------------------
    def update_sl_tp(
        self,
        position_id: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> None:
        pos = self._positions.get(position_id)
        if pos is None or not pos.is_open:
            return

        inst = pos.symbol.replace("/", "-")

        # Cancelar órdenes reduceOnly previas
        existing = self.trade.get_order_list(instId=inst)
        if "data" in existing:
            for o in existing["data"]:
                if o.get("reduceOnly") == "true":
                    try:
                        self.trade.cancel_order(instId=inst, ordId=o["ordId"])
                    except Exception:
                        pass

        if sl is not None:
            pos.sl = sl
        if tp is not None:
            pos.tp = tp

        self._submit_sl_tp(pos.symbol, pos.side, pos.size, pos.sl, pos.tp)

    # -------------------------- close_position -----------------------
    def close_position(self, position_id: str, reason: str) -> None:
        pos = self._positions.get(position_id)
        if pos is None or not pos.is_open:
            return

        inst = pos.symbol.replace("/", "-")
        close_side = "sell" if pos.side == "long" else "buy"

        try:
            self.trade.place_order(
                instId=inst,
                tdMode="cross",
                side=close_side,
                ordType="market",
                sz=str(pos.size),
                reduceOnly=True,
            )
        except Exception:
            pass

        pos.is_open = False
        pos.close_time = datetime.utcnow()
        pos.close_reason = reason

    # ----------------------------- now() -----------------------------
    def now(self) -> datetime:
        return datetime.utcnow()