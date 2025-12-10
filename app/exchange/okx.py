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
from typing import Dict, Optional, Any, Iterator, List
from datetime import datetime, timedelta, timezone
import time
import pandas as pd

from app.exchange.base import IExchange

try:
    from okx.Account_api import AccountAPI
    from okx.Market_api import MarketAPI
    from okx.Trade_api import TradeAPI
except Exception:
    raise ImportError("Falta instalar el SDK oficial de OKX: pip install okx-python")


# Debug local para este módulo
DEBUG_OKX = False


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
        """
        Implementación pendiente. Podés reutilizar la misma lógica de
        _fetch_ohlcv construyendo un DataFrame.
        """
        raise NotImplementedError

    def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Wrapper interno para traer velas desde OKX.

        Devuelve lista de dicts:
            - "timestamp": pd.Timestamp (UTC)
            - "open", "high", "low", "close": float
        """
        inst = symbol.replace("/", "-")
        params: Dict[str, Any] = {
            "instId": inst,
            "bar": timeframe,
            "limit": str(limit),
        }
        # OKX usa before/after; acá usamos 'before' como inicio aproximado.
        # Chequear doc oficial y ajustar según sea necesario.
        params["before"] = str(since_ms)

        data = self.market.get_history_candlesticks(**params)
        if "data" not in data or not data["data"]:
            return []

        rows: List[Dict[str, Any]] = []
        # La API devuelve velas en orden descendente; las ordenamos luego.
        for ohlc in data["data"]:
            # ohlc[0] = ts ms, [1]=open, [2]=high, [3]=low, [4]=close, [5]=vol
            ts_ms = int(ohlc[0])
            ts = pd.to_datetime(ts_ms, unit="ms", utc=True)
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

        rows = sorted(rows, key=lambda r: r["timestamp"])
        return rows

    def iter_position_bars(
        self,
        symbol: str,
        price_tf: str,
        ema_tf: str,
        start_time: pd.Timestamp,
        max_minutes: int,
        ema_span: int,
    ) -> Iterator[Dict[str, Any]]:
        """
        Itera barras de precio en vivo desde OKX.

        - Arranca en start_time.
        - Sigue pidiendo nuevas velas mientras:
              now < start_time + max_minutes
        - Calcula EMA incrementalmente en ema_tf.
        """
        if not isinstance(start_time, pd.Timestamp):
            start_time = pd.to_datetime(start_time)

        end_time = start_time + timedelta(minutes=max_minutes)

        # Estado EMA (TF de EMA puede ser igual o distinto a price_tf)
        ema_values: List[Dict[str, Any]] = []

        # Trabajamos en ms desde epoch (UTC)
        current_from = int(start_time.replace(tzinfo=timezone.utc).timestamp() * 1000)

        # Mapping TF->segundos (ajustar si usás otros)
        tf_to_secs = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400,
            "1w": 604800,
        }
        step_secs = tf_to_secs.get(price_tf, 60)

        last_bar_ts: Optional[pd.Timestamp] = None

        while True:
            now = datetime.now(timezone.utc)
            if now >= end_time.replace(tzinfo=timezone.utc):
                break

            # Traer velas nuevas de precio
            try:
                ohlcv = self._fetch_ohlcv(
                    symbol=symbol,
                    timeframe=price_tf,
                    since_ms=current_from,
                    limit=200,
                )
            except Exception as e:
                if DEBUG_OKX:
                    print(f"[OKX] Error fetch_ohlcv price: {e}")
                time.sleep(1)
                continue

            if not ohlcv:
                # No hay velas nuevas todavía, esperamos un poco
                time.sleep(step_secs / 2)
                continue

            # Procesar velas en orden cronológico
            for bar in ohlcv:
                t = bar["timestamp"]
                if not isinstance(t, pd.Timestamp):
                    t = pd.to_datetime(t, unit="ms", utc=True)

                if t < start_time:
                    continue
                if t > end_time:
                    break

                h = float(bar.get("high", bar["close"]))
                l = float(bar.get("low", bar["close"]))
                c = float(bar["close"])

                # Actualizar estado de EMA en TF de EMA
                # (para simplificar, usamos el mismo flujo con ema_tf == price_tf)
                ema_values.append({"timestamp": t, "close": c})
                df_ema = pd.DataFrame(ema_values).sort_values("timestamp")
                df_ema["ema"] = df_ema["close"].ewm(
                    span=ema_span,
                    adjust=False,
                ).mean()

                ema_row = df_ema[df_ema["timestamp"] <= t].tail(1)
                if not ema_row.empty and not pd.isna(ema_row["ema"].iloc[0]):
                    ema_val = float(ema_row["ema"].iloc[0])
                else:
                    ema_val = c

                last_bar_ts = t
                yield {
                    "timestamp": t,
                    "high": h,
                    "low": l,
                    "close": c,
                    "ema": ema_val,
                }

            # Avanzar "since" para la próxima llamada
            if last_bar_ts is not None:
                current_from = int(last_bar_ts.timestamp() * 1000) + 1

            # Esperar un poco antes de la próxima consulta
            time.sleep(step_secs / 2)

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