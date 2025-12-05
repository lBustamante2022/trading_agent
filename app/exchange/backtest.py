# app/exchange/backtest.py
"""
Implementación de Exchange para BACKTEST en GREEN v3.

Esta clase implementa la interfaz IExchange usando:
    - Velas históricas ya cargadas en memoria (dict[symbol][timeframe] -> DataFrame)
    - Posiciones "virtuales" en memoria (no toca ningún broker real)

NO contiene ninguna lógica propia de GREEN (impulsos, pullbacks, etc).
Solo sabe:
    - Devolver candles (OHLCV) por símbolo / timeframe.
    - Crear / actualizar / cerrar posiciones virtuales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from datetime import datetime

import pandas as pd

from app.exchange.base import IExchange


@dataclass
class BacktestPosition:
    """
    Representa una posición virtual en un backtest.
    No calcula PnL ni RR; eso lo hace la lógica de GREEN/Position.
    """
    position_id: str
    symbol: str
    side: str          # "long" o "short"
    size: float
    entry_price: float
    sl: float
    tp: float
    open_time: datetime
    is_open: bool = True
    close_time: Optional[datetime] = None
    close_reason: Optional[str] = None


class BacktestExchange(IExchange):
    """
    Implementación de IExchange para backtests.

    candles_by_symbol:
        {
            "BTC/USDT": {
                "1m":  df_1m,
                "3m":  df_3m,
                "15m": df_15m,
                ...
            },
            "ETH/USDT": {
                ...
            },
            ...
        }

    Cada DataFrame debe tener:
        - timestamp (pd.Timestamp)
        - open, high, low, close (float)
        - volume (opcional)
    """

    def __init__(
        self,
        candles_by_symbol: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
        clock_start: Optional[datetime] = None,
    ) -> None:
        # Permite usarlo con o sin pasarle velas (por compatibilidad con run_backtest)
        self._candles_by_symbol: Dict[str, Dict[str, pd.DataFrame]] = candles_by_symbol or {}
        self._positions: Dict[str, BacktestPosition] = {}
        self._now: datetime = clock_start or datetime.utcnow()

    # ------------------------------------------------------------------
    # Utilidad interna: normalizar timestamp
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_ts_index(df: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df = df.copy()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    # ------------------------------------------------------------------
    # IExchange: datos de mercado
    # ------------------------------------------------------------------
    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        symbol_maps = self._candles_by_symbol.get(symbol)
        if symbol_maps is None:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = symbol_maps.get(timeframe)
        if df is None or df.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = self._ensure_ts_index(df).sort_values("timestamp")

        if start is not None:
            df = df[df["timestamp"] >= pd.to_datetime(start)]
        if end is not None:
            df = df[df["timestamp"] <= pd.to_datetime(end)]

        return df.reset_index(drop=True).copy()

    # ------------------------------------------------------------------
    # IExchange: gestión de posiciones
    # ------------------------------------------------------------------
    def create_position(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        sl: float,
        tp: float,
    ) -> str:
        now = self.now()
        position_id = f"{symbol}-{side}-{len(self._positions) + 1}-{int(now.timestamp())}"

        pos = BacktestPosition(
            position_id=position_id,
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            open_time=now,
        )
        self._positions[position_id] = pos
        return position_id

    def update_sl_tp(
        self,
        position_id: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> None:
        pos = self._positions.get(position_id)
        if pos is None or not pos.is_open:
            return

        if sl is not None:
            pos.sl = sl
        if tp is not None:
            pos.tp = tp

    def close_position(
        self,
        position_id: str,
        reason: str,
    ) -> None:
        pos = self._positions.get(position_id)
        if pos is None or not pos.is_open:
            return

        pos.is_open = False
        pos.close_time = self.now()
        pos.close_reason = reason

    # ------------------------------------------------------------------
    # IExchange: tiempo
    # ------------------------------------------------------------------
    def now(self) -> datetime:
        return self._now

    def set_now(self, new_now: datetime) -> None:
        self._now = new_now