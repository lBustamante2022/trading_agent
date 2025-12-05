# app/green/pipeline/position.py
"""
Gestión de posición para GREEN V3 (arquitectura nueva).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import builtins as _builtins

import pandas as pd
import numpy as np

from app.green.core import Entry, Impulse, TradeResult
from app.green.styles import GreenStyle
from app.exchange.base import IExchange


# ----------------------------------------------------------------------
# DEBUG LOCAL
# ----------------------------------------------------------------------

DEBUG_POSITION = False


def _log(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = " ".join(str(a) for a in args)
    _builtins.print(f"[{ts}] {msg}", **kwargs)


print = _log  # override local


# ----------------------------------------------------------------------
# Resultado interno de gestión
# ----------------------------------------------------------------------

@dataclass
class PositionResult:
    exit_time: pd.Timestamp
    exit_price: float
    result: str      # "win" | "loss" | "be"
    rr_real: float   # R realizado


# ----------------------------------------------------------------------
# Motor de gestión (TP1 / TP2 / BE / EMA + timeout)
# ----------------------------------------------------------------------

def _simulate_position(
    direction: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    sl_price: float,
    tp_price: float,
    df_price: pd.DataFrame,
    df_ema_tf: pd.DataFrame,
    ema_trail_buffer_pct: float,
    ema_span: int,
    max_holding_minutes: int,
    exchange: Optional[IExchange] = None,
    position_id: Optional[str] = None,
) -> Optional[PositionResult]:
    if df_price is None or df_price.empty:
        return None
    if df_ema_tf is None or df_ema_tf.empty:
        return None

    # 1) EMA en TF de trailing
    df_ema = df_ema_tf.copy().sort_values("timestamp")
    df_ema["ema"] = df_ema["close"].ewm(span=ema_span, adjust=False).mean()

    # 2) Ventana de vida
    end_time = entry_time + timedelta(minutes=max_holding_minutes)
    df_price = df_price[
        (df_price["timestamp"] >= entry_time) &
        (df_price["timestamp"] <= end_time)
    ].copy()
    df_price = df_price.sort_values("timestamp").reset_index(drop=True)
    if df_price.empty:
        return None

    # 3) TP1 / TP2
    if direction == "long":
        zone = tp_price - entry_price
        if zone <= 0:
            return None
        tp1 = entry_price + zone / 3.0
        tp2 = entry_price + 2.0 * zone / 3.0
    else:
        zone = entry_price - tp_price
        if zone <= 0:
            return None
        tp1 = entry_price - zone / 3.0
        tp2 = entry_price - 2.0 * zone / 3.0

    orig_sl = sl_price
    sl = sl_price
    state = "initial"   # "initial" → "after_tp1" → "after_tp2"

    # 1R
    if direction == "long":
        one_r = entry_price - orig_sl
    else:
        one_r = orig_sl - entry_price
    if one_r <= 0:
        return None

    # 4) Vela a vela
    for _, row in df_price.iterrows():
        t = row["timestamp"]
        h = float(row.get("high", row["close"]))
        l = float(row.get("low", row["close"]))

        # Trailing EMA
        if state == "after_tp2":
            ema_row = df_ema[df_ema["timestamp"] <= t].tail(1)
            if not ema_row.empty:
                ema = float(ema_row["ema"].iloc[0])
                buffer = ema * ema_trail_buffer_pct
                if direction == "long":
                    trail_sl = ema - buffer
                    sl = max(sl, trail_sl)
                else:
                    trail_sl = ema + buffer
                    sl = min(sl, trail_sl)

                if exchange is not None and position_id is not None:
                    try:
                        exchange.update_sl_tp(position_id, sl=sl, tp=tp_price)
                    except Exception as e:
                        if DEBUG_POSITION:
                            print(f"[POSITION] Error update_sl_tp trailing: {e}")

        # Check SL / TP
        if direction == "long":
            # SL
            if l <= sl:
                exit_price = sl
                if exchange is not None and position_id is not None:
                    try:
                        exchange.close_position(position_id, reason="sl_hit")
                    except Exception as e:
                        if DEBUG_POSITION:
                            print(f"[POSITION] Error close_position SL: {e}")

                if np.isclose(exit_price, entry_price) or exit_price > entry_price:
                    result = "be"
                    rr_real = 0.0
                else:
                    result = "loss"
                    rr_real = (exit_price - entry_price) / one_r
                return PositionResult(t, exit_price, result, rr_real)

            # TP1 / TP2 / TP
            if state == "initial" and h >= tp1:
                sl = entry_price - 0.5 * (entry_price - orig_sl)
                state = "after_tp1"
                if exchange is not None and position_id is not None:
                    try:
                        exchange.update_sl_tp(position_id, sl=sl, tp=tp_price)
                    except Exception as e:
                        if DEBUG_POSITION:
                            print(f"[POSITION] Error update_sl_tp TP1: {e}")

            if state in ("initial", "after_tp1") and h >= tp2:
                sl = entry_price
                state = "after_tp2"
                if exchange is not None and position_id is not None:
                    try:
                        exchange.update_sl_tp(position_id, sl=sl, tp=tp_price)
                    except Exception as e:
                        if DEBUG_POSITION:
                            print(f"[POSITION] Error update_sl_tp TP2: {e}")

            if h >= tp_price:
                exit_price = tp_price
                if exchange is not None and position_id is not None:
                    try:
                        exchange.close_position(position_id, reason="tp_hit")
                    except Exception as e:
                        if DEBUG_POSITION:
                            print(f"[POSITION] Error close_position TP: {e}")

                result = "win"
                rr_real = (exit_price - entry_price) / one_r
                return PositionResult(t, exit_price, result, rr_real)

        else:  # short
            # SL
            if h >= sl:
                exit_price = sl
                if exchange is not None and position_id is not None:
                    try:
                        exchange.close_position(position_id, reason="sl_hit")
                    except Exception as e:
                        if DEBUG_POSITION:
                            print(f"[POSITION] Error close_position SL: {e}")

                if np.isclose(exit_price, entry_price) or exit_price < entry_price:
                    result = "be"
                    rr_real = 0.0
                else:
                    result = "loss"
                    rr_real = (entry_price - exit_price) / one_r
                return PositionResult(t, exit_price, result, rr_real)

            # TP1 / TP2 / TP
            if state == "initial" and l <= tp1:
                sl = entry_price + 0.5 * (orig_sl - entry_price)
                state = "after_tp1"
                if exchange is not None and position_id is not None:
                    try:
                        exchange.update_sl_tp(position_id, sl=sl, tp=tp_price)
                    except Exception as e:
                        if DEBUG_POSITION:
                            print(f"[POSITION] Error update_sl_tp TP1: {e}")

            if state in ("initial", "after_tp1") and l <= tp2:
                sl = entry_price
                state = "after_tp2"
                if exchange is not None and position_id is not None:
                    try:
                        exchange.update_sl_tp(position_id, sl=sl, tp=tp_price)
                    except Exception as e:
                        if DEBUG_POSITION:
                            print(f"[POSITION] Error update_sl_tp TP2: {e}")

            if l <= tp_price:
                exit_price = tp_price
                if exchange is not None and position_id is not None:
                    try:
                        exchange.close_position(position_id, reason="tp_hit")
                    except Exception as e:
                        if DEBUG_POSITION:
                            print(f"[POSITION] Error close_position TP: {e}")

                result = "win"
                rr_real = (entry_price - exit_price) / one_r
                return PositionResult(t, exit_price, result, rr_real)

    # 5) Timeout
    last = df_price.iloc[-1]
    t = last["timestamp"]
    exit_price = float(last["close"])

    if exchange is not None and position_id is not None:
        try:
            exchange.close_position(position_id, reason="timeout")
        except Exception as e:
            if DEBUG_POSITION:
                print(f"[POSITION] Error close_position timeout: {e}")

    if direction == "long":
        rr_real = (exit_price - entry_price) / one_r
    else:
        rr_real = (entry_price - exit_price) / one_r

    if rr_real > 0:
        result = "win"
    elif np.isclose(rr_real, 0.0):
        result = "be"
        rr_real = 0.0
    else:
        result = "loss"

    return PositionResult(t, exit_price, result, rr_real)


# ----------------------------------------------------------------------
# Estrategia de posición para el pipeline GREEN V3
# ----------------------------------------------------------------------

@dataclass
class DefaultPositionStrategy:
    """
    Estrategia de posición genérica para GREEN V3.

    Usa:
        - style.position_price_tf
        - style.position_ema_tf
        - style.max_holding_minutes

        - cfg.ema_trail_buffer_pct (default 0.002)
        - cfg.ema_trail_span (default 50)
        - cfg.live_position_size (opcional, para LIVE con Exchange)
    """

    def simulate_trade(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        entry: Entry,
        style: GreenStyle,
        cfg: Any,
        exchange: IExchange,
    ) -> Optional[TradeResult]:
        tf_price = style.position_price_tf
        tf_ema = style.position_ema_tf

        df_price = dfs.get(tf_price)
        df_ema = dfs.get(tf_ema)

        if df_price is None or df_price.empty or df_ema is None or df_ema.empty:
            return None

        df_price = df_price.sort_values("timestamp").reset_index(drop=True)
        df_ema = df_ema.sort_values("timestamp").reset_index(drop=True)

        # Precio de entrada = close de la vela entry_time
        row_entry = df_price[df_price["timestamp"] == entry.entry_time]
        if row_entry.empty:
            return None

        entry_price = float(row_entry["close"].iloc[0])
        sl = float(entry.sl)
        tp = float(entry.tp)

        # RR planificado
        if entry.direction == "long":
            one_r = entry_price - sl
            if one_r <= 0:
                return None
            rr_planned = (tp - entry_price) / one_r
        else:
            one_r = sl - entry_price
            if one_r <= 0:
                return None
            rr_planned = (entry_price - tp) / one_r

        max_holding_minutes = getattr(style, "max_holding_minutes", 24 * 60)
        ema_trail_buffer_pct = float(getattr(cfg, "ema_trail_buffer_pct", 0.002))
        ema_span = int(getattr(cfg, "ema_trail_span", 50))

        # Crear posición real si corresponde
        position_id: Optional[str] = None
        size = float(getattr(cfg, "live_position_size", 0.0))
        if size > 0:
            side = "long" if entry.direction == "long" else "short"
            try:
                position_id = exchange.create_position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    sl=sl,
                    tp=tp,
                )
            except Exception as e:
                if DEBUG_POSITION:
                    print(f"[POSITION] Error create_position en exchange: {e}")
                position_id = None

        pos = _simulate_position(
            direction=entry.direction,
            entry_time=entry.entry_time,
            entry_price=entry_price,
            sl_price=sl,
            tp_price=tp,
            df_price=df_price,
            df_ema_tf=df_ema,
            ema_trail_buffer_pct=ema_trail_buffer_pct,
            ema_span=ema_span,
            max_holding_minutes=max_holding_minutes,
            exchange=exchange,
            position_id=position_id,
        )

        if pos is None:
            return None

        impulse_start = getattr(entry.impulse, "start", getattr(entry.impulse, "start_time", None))

        return TradeResult(
            symbol=symbol,
            direction=entry.direction,
            entry_time=entry.entry_time,
            entry_price=entry_price,
            sl_initial=sl,
            tp_initial=tp,
            exit_time=pos.exit_time,
            exit_price=pos.exit_price,
            result=pos.result,
            rr_planned=rr_planned,
            rr_real=pos.rr_real,
            trigger_time=entry.trigger.timestamp,
            impulse_start=impulse_start,
            meta={},
        )