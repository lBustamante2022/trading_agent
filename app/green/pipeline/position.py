"""
Gestión de posición para GREEN V3 (arquitectura nueva, revisada).

Cambios clave:

    - simulate_trade → gestion_trade (nombre solicitado).
    - Usa exchange.iter_position_bars(...) para avanzar vela a vela
      tanto en backtest como en live.
    - Mantiene lógica de:
        * TP1 → mover SL.
        * TP2 → SL a BE y activar trailing EMA.
        * SL / TP / timeout → cierre posición.
    - NO hace TP parcial (solo trailing vía SL).
    - Corrige clasificación de resultado:
        * RR_real > 0  → "win"
        * RR_real ≈ 0  → "be"
        * RR_real < 0  → "loss"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
import builtins as _builtins

import pandas as pd
import numpy as np

from app.green.core import Entry, TradeResult
from app.green.styles import GreenStyle
from app.exchange.base import IExchange

# ----------------------------------------------------------------------
# DEBUG LOCAL
# ----------------------------------------------------------------------

DEBUG_POSITION = False


def _log(*args, **kwargs):
    if DEBUG_POSITION:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = " ".join(str(a) for a in args)
        _builtins.print(f"[{ts} DEBUG POSITION] {msg}", **kwargs)


print = _log  # override local


# ----------------------------------------------------------------------
# Estrategia de posición
# ----------------------------------------------------------------------

@dataclass
class DefaultPositionStrategy:
    """
    Estrategia de posición genérica para GREEN V3.

    Usa:
        - style.position_price_tf
        - style.position_ema_tf
        - style.max_holding_minutes (o entry_max_minutes como fallback)

        - cfg.position_ema_trail_buffer_pct (default 0.002)
        - cfg.position_ema_trail_span (default 50)
        - cfg.position_size (opcional, para LIVE con Exchange)
    """

    exchange: IExchange

    def gestion_trade(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        entry: Entry,
        style: GreenStyle,
        cfg: Any,
    ) -> Optional[TradeResult]:

        direction = entry.meta["direction"]  # "long" / "short"
        entry_time = entry.meta["end"]
        if not isinstance(entry_time, pd.Timestamp):
            entry_time = pd.to_datetime(entry_time)

        tf_price = style.position_price_tf
        tf_ema = style.position_ema_tf

        # DataFrames de referencia (para backtest o validaciones)
        df_price = dfs.get(tf_price)
        df_ema = dfs.get(tf_ema)

        if df_price is None or df_price.empty:
            return None
        if df_ema is None or df_ema.empty:
            # fallback: usar df_price para EMA si hiciera falta
            df_ema = df_price

        df_price = df_price.sort_values("timestamp").reset_index(drop=True)
        df_ema = df_ema.sort_values("timestamp").reset_index(drop=True)

        # Precio de entrada:
        #  - Usamos preferentemente el "entry" que calculó Entry.
        #  - Fallback: close de la vela de entry_time.
        entry_price = float(entry.meta.get("entry", np.nan))
        if np.isnan(entry_price):
            row_entry = df_price[df_price["timestamp"] == entry_time]
            if row_entry.empty:
                return None
            entry_price = float(row_entry["close"].iloc[0])

        sl = float(entry.meta["sl"])
        tp = float(entry.meta["tp"])
        orig_sl = sl

        # 1R planificado y RR objetivo (informativo)
        if direction == "long":
            one_r = entry_price - sl
            if one_r <= 0:
                return None
            rr_planned = (tp - entry_price) / one_r
        else:
            one_r = sl - entry_price
            if one_r <= 0:
                return None
            rr_planned = (entry_price - tp) / one_r

        # Tiempo máximo de vida de la posición
        max_holding_minutes = int(
            getattr(
                style,
                "max_holding_minutes",
                getattr(style, "entry_max_minutes", 24 * 60),
            )
        )

        ema_trail_buffer_pct = float(
            getattr(cfg, "position_ema_trail_buffer_pct", 0.002)
        )
        ema_span = int(getattr(cfg, "position_ema_trail_span", 50))

        # --------------------------------------------------------------
        # Crear posición real si corresponde (live)
        # --------------------------------------------------------------
        position_id: Optional[str] = None
        size = float(getattr(cfg, "position_size", 0.0))
        if size > 0:
            side = "long" if direction == "long" else "short"
            try:
                position_id = self.exchange.create_position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    sl=sl,
                    tp=tp,
                )
            except Exception as e:
                if DEBUG_POSITION:
                    print(f"Error create_position en exchange: {e}")
                position_id = None

        # --------------------------------------------------------------
        # Iterador de barras desde el Exchange
        #   - Backtest: recorre DF interno.
        #   - Live: trae velas reales (REST/WebSocket).
        # --------------------------------------------------------------
        bars_iter = self.exchange.iter_position_bars(
            symbol=symbol,
            price_tf=tf_price,
            ema_tf=tf_ema,
            start_time=entry_time,
            max_minutes=max_holding_minutes,
            ema_span=ema_span,
        )

        # Estados de gestión
        state = "initial"   # "initial" → "after_tp1" → "after_tp2"

        # Precalcular TP1 / TP2 en términos de precio
        if direction == "long":
            zone = tp - entry_price
            if zone <= 0:
                return None
            tp1 = entry_price + zone / 3.0
            tp2 = entry_price + 2.0 * zone / 3.0
        else:
            zone = entry_price - tp
            if zone <= 0:
                return None
            tp1 = entry_price - zone / 3.0
            tp2 = entry_price - 2.0 * zone / 3.0

        last_bar_time: Optional[pd.Timestamp] = None
        last_close: Optional[float] = None

        # --------------------------------------------------------------
        # LOOP vela a vela
        # --------------------------------------------------------------
        for bar in bars_iter:
            t = bar["timestamp"]
            if not isinstance(t, pd.Timestamp):
                t = pd.to_datetime(t)

            h = float(bar.get("high", bar["close"]))
            l = float(bar.get("low", bar["close"]))
            c = float(bar["close"])
            ema_val = float(bar["ema"])

            last_bar_time = t
            last_close = c

            # Trailing EMA (solo después de TP2)
            if state == "after_tp2":
                buffer = ema_val * ema_trail_buffer_pct
                if direction == "long":
                    trail_sl = ema_val - buffer
                    sl = max(sl, trail_sl)
                else:
                    trail_sl = ema_val + buffer
                    sl = min(sl, trail_sl)

                if position_id is not None:
                    try:
                        self.exchange.update_sl_tp(position_id, sl=sl, tp=tp)
                    except Exception as e:
                        if DEBUG_POSITION:
                            print(f"Error update_sl_tp trailing: {e}")

            # ----------------------------------------------------------
            # Check de SL / TP1 / TP2 / TP
            # ----------------------------------------------------------
            if direction == "long":
                # SL
                if l <= sl:
                    exit_price = sl
                    reason = "sl_hit"

                    if position_id is not None:
                        try:
                            self.exchange.close_position(position_id, reason=reason)
                        except Exception as e:
                            if DEBUG_POSITION:
                                print(f"Error close_position SL: {e}")

                    rr_real = (exit_price - entry_price) / one_r
                    result = self._classify_result(rr_real)
                    return self._build_trade_result(
                        symbol=symbol,
                        entry=entry,
                        entry_price=entry_price,
                        sl_initial=orig_sl,
                        tp_initial=tp,
                        exit_time=t,
                        exit_price=exit_price,
                        result=result,
                        rr_planned=rr_planned,
                        rr_real=rr_real,
                    )

                # TP1
                if state == "initial" and h >= tp1:
                    # subimos SL a mitad del camino entre entry y SL original
                    sl = entry_price - 0.5 * (entry_price - orig_sl)
                    state = "after_tp1"
                    if position_id is not None:
                        try:
                            self.exchange.update_sl_tp(position_id, sl=sl, tp=tp)
                        except Exception as e:
                            if DEBUG_POSITION:
                                print(f"Error update_sl_tp TP1: {e}")

                # TP2
                if state in ("initial", "after_tp1") and h >= tp2:
                    sl = entry_price  # SL a BE
                    state = "after_tp2"
                    if position_id is not None:
                        try:
                            self.exchange.update_sl_tp(position_id, sl=sl, tp=tp)
                        except Exception as e:
                            if DEBUG_POSITION:
                                print(f"Error update_sl_tp TP2: {e}")

                # TP final
                if h >= tp:
                    exit_price = tp
                    reason = "tp_hit"

                    if position_id is not None:
                        try:
                            self.exchange.close_position(position_id, reason=reason)
                        except Exception as e:
                            if DEBUG_POSITION:
                                print(f"Error close_position TP: {e}")

                    rr_real = (exit_price - entry_price) / one_r
                    result = self._classify_result(rr_real)
                    return self._build_trade_result(
                        symbol=symbol,
                        entry=entry,
                        entry_price=entry_price,
                        sl_initial=orig_sl,
                        tp_initial=tp,
                        exit_time=t,
                        exit_price=exit_price,
                        result=result,
                        rr_planned=rr_planned,
                        rr_real=rr_real,
                    )

            else:  # SHORT
                # SL
                if h >= sl:
                    exit_price = sl
                    reason = "sl_hit"

                    if position_id is not None:
                        try:
                            self.exchange.close_position(position_id, reason=reason)
                        except Exception as e:
                            if DEBUG_POSITION:
                                print(f"Error close_position SL: {e}")

                    rr_real = (entry_price - exit_price) / one_r
                    result = self._classify_result(rr_real)
                    return self._build_trade_result(
                        symbol=symbol,
                        entry=entry,
                        entry_price=entry_price,
                        sl_initial=orig_sl,
                        tp_initial=tp,
                        exit_time=t,
                        exit_price=exit_price,
                        result=result,
                        rr_planned=rr_planned,
                        rr_real=rr_real,
                    )

                # TP1
                if state == "initial" and l <= tp1:
                    sl = entry_price + 0.5 * (orig_sl - entry_price)
                    state = "after_tp1"
                    if position_id is not None:
                        try:
                            self.exchange.update_sl_tp(position_id, sl=sl, tp=tp)
                        except Exception as e:
                            if DEBUG_POSITION:
                                print(f"Error update_sl_tp TP1: {e}")

                # TP2
                if state in ("initial", "after_tp1") and l <= tp2:
                    sl = entry_price  # SL a BE
                    state = "after_tp2"
                    if position_id is not None:
                        try:
                            self.exchange.update_sl_tp(position_id, sl=sl, tp=tp)
                        except Exception as e:
                            if DEBUG_POSITION:
                                print(f"Error update_sl_tp TP2: {e}")

                # TP final
                if l <= tp:
                    exit_price = tp
                    reason = "tp_hit"

                    if position_id is not None:
                        try:
                            self.exchange.close_position(position_id, reason=reason)
                        except Exception as e:
                            if DEBUG_POSITION:
                                print(f"Error close_position TP: {e}")

                    rr_real = (entry_price - exit_price) / one_r
                    result = self._classify_result(rr_real)
                    return self._build_trade_result(
                        symbol=symbol,
                        entry=entry,
                        entry_price=entry_price,
                        sl_initial=orig_sl,
                        tp_initial=tp,
                        exit_time=t,
                        exit_price=exit_price,
                        result=result,
                        rr_planned=rr_planned,
                        rr_real=rr_real,
                    )

        # --------------------------------------------------------------
        # TIMEOUT: si el iterador terminó sin tocar SL/TP
        # --------------------------------------------------------------
        if last_bar_time is None or last_close is None:
            # No hubo ni una barra tras la entrada
            return None

        exit_time = last_bar_time
        exit_price = float(last_close)

        if position_id is not None:
            try:
                self.exchange.close_position(position_id, reason="timeout")
            except Exception as e:
                if DEBUG_POSITION:
                    print(f"Error close_position timeout: {e}")

        if direction == "long":
            rr_real = (exit_price - entry_price) / one_r
        else:
            rr_real = (entry_price - exit_price) / one_r

        result = self._classify_result(rr_real)

        return self._build_trade_result(
            symbol=symbol,
            entry=entry,
            entry_price=entry_price,
            sl_initial=orig_sl,
            tp_initial=tp,
            exit_time=exit_time,
            exit_price=exit_price,
            result=result,
            rr_planned=rr_planned,
            rr_real=rr_real,
        )

    # ------------------------------------------------------------------
    # Helper: clasificar resultado según RR realizado
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_result(rr_real: float) -> str:
        if np.isclose(rr_real, 0.0, atol=1e-6):
            return "be"
        if rr_real > 0:
            return "win"
        return "loss"

    # ------------------------------------------------------------------
    # Helper: construir TradeResult coherente con el core
    # ------------------------------------------------------------------
    @staticmethod
    def _build_trade_result(
        symbol: str,
        entry: Entry,
        entry_price: float,
        sl_initial: float,
        tp_initial: float,
        exit_time: pd.Timestamp,
        exit_price: float,
        result: str,
        rr_planned: float,
        rr_real: float,
    ) -> TradeResult:

        direction = entry.meta["direction"]

        trigger_time = entry.trigger.meta.get("end")
        if not isinstance(trigger_time, pd.Timestamp):
            trigger_time = pd.to_datetime(trigger_time)

        impulse = entry.trigger.pullback.impulse
        # En la nueva arquitectura, los tiempos están en impulse.meta
        impulse_start = impulse.meta.get("start")
        if impulse_start is not None and not isinstance(impulse_start, pd.Timestamp):
            impulse_start = pd.to_datetime(impulse_start)

        entry_time = entry.meta.get("end")
        if not isinstance(entry_time, pd.Timestamp):
            entry_time = pd.to_datetime(entry_time)

        return TradeResult(
            symbol=symbol,
            direction=direction,
            entry_time=entry_time,
            entry_price=entry_price,
            sl_initial=sl_initial,
            tp_initial=tp_initial,
            exit_time=exit_time,
            exit_price=exit_price,
            result=result,
            rr_planned=rr_planned,
            rr_real=rr_real,
            trigger_time=trigger_time,
            impulse_start=impulse_start,
        )