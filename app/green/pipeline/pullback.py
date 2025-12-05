# app/green/pipeline/pullback.py
"""
Detección de PULLBACK para GREEN v3 (arquitectura nueva).

Idea (igual que la versión anterior, pero genérica):

    SHORT (impulso bajista desde resistencia):
        - En el TF de pullback (style.pullback_tf) buscamos una serie
          de swing highs:
              * cerca del nivel SR del impulso
              * con máximos crecientes (cuña alcista)
              * duración mínima en horas
              * el último swing y su cierre cerca del SR
        → Pullback alcista agotándose contra resistencia.

    LONG (impulso alcista desde soporte):
        - En el TF de pullback buscamos swing lows:
              * cerca del SR del impulso
              * mínimos decrecientes (cuña bajista)
              * duración mínima
              * último swing + cierre cerca del SR
        → Pullback bajista agotándose contra soporte.

No asume timeframes fijos, toma el TF de style.pullback_tf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from datetime import datetime, timedelta

import builtins as _builtins
import pandas as pd

from app.green.core import PullbackStrategy, Pullback, Impulse
from app.green.styles import GreenStyle
from app.ta.swing_points import find_swing_highs, find_swing_lows


# ----------------------------------------------------------------------
# DEBUG PRINT LOCAL (opcional)
# ----------------------------------------------------------------------

DEBUG_PULLBACK = False


def _log_with_ts(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = " ".join(str(a) for a in args)
    _builtins.print(f"[{ts}] {msg}", **kwargs)


print = _log_with_ts  # solo afecta a este archivo


def _debug_swings(label: str, swings: List[Tuple[pd.Timestamp, float]], sr: float):
    if not DEBUG_PULLBACK:
        return
    print(f"[PULLBACK DEBUG] {label}: {len(swings)} swings")
    for ts, price in swings:
        dist = abs(price - sr) / sr
        print(f"   - {ts}  price={price:.2f}  dist={dist:.4f}")


# ----------------------------------------------------------------------
# Estrategia concreta de Pullback
# ----------------------------------------------------------------------

@dataclass
class DefaultPullbackStrategy(PullbackStrategy):
    """
    Estrategia genérica de pullback en cuña contra SR.

    Parámetros usados:

      De cfg:
        - pullback_min_swings (int)
        - pullback_max_days (float)
        - pullback_sr_tolerance (float)  ej: 0.02 = 2%

      Del style:
        - pullback_tf (str)
        - pullback_min_hours (float, opcional; default 12h)

    Devuelve: lista de Pullback (en esta versión, 0 o 1 por impulso).
    """

    def detect_pullbacks(
        self,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        impulse: Impulse,
    ) -> List[Pullback]:

        tf = style.pullback_tf
        df = dfs.get(tf)
        if df is None or df.empty:
            return []

        if "timestamp" not in df.columns or "high" not in df.columns or "low" not in df.columns:
            return []

        df = df.sort_values("timestamp").reset_index(drop=True)

        # --------------------------------------------------------------
        # Ventana temporal después del impulso
        # --------------------------------------------------------------
        max_days = float(getattr(cfg, "pullback_max_days", 10.0))
        start_ts = impulse.start
        end_ts = impulse.start + timedelta(days=max_days)

        df_win = df[
            (df["timestamp"] > start_ts) &
            (df["timestamp"] <= end_ts)
        ].copy()

        if df_win.empty:
            if DEBUG_PULLBACK:
                print(f"[PULLBACK DEBUG] ventana vacía ({start_ts} → {end_ts})")
            return []

        sr_level = float(impulse.sr_level)
        direction = impulse.direction

        min_swings = max(2, int(getattr(cfg, "pullback_min_swings", 2)))
        tol = float(getattr(cfg, "pullback_sr_tolerance", 0.02))     # tolerancia base al SR
        close_tol = tol * 1.5                                        # un poco más para el cierre

        min_hours = float(getattr(style, "pullback_min_hours", 12.0))

        if DEBUG_PULLBACK:
            print(
                f"[PULLBACK DEBUG] dir={direction}, tf={tf}, "
                f"min_swings={min_swings}, max_days={max_days}, "
                f"tol={tol:.3f}, close_tol={close_tol:.3f}, "
                f"sr_level={sr_level:.2f}, min_hours={min_hours:.1f}"
            )

        # --------------------------------------------------------------
        # SHORT → pullback alcista contra resistencia (swing highs)
        # --------------------------------------------------------------
        if direction == "short":
            highs = df_win["high"].to_numpy()
            idx_sw = find_swing_highs(highs)

            if len(idx_sw) < min_swings:
                return []

            swings_all: List[Tuple[pd.Timestamp, float, float]] = []
            for idx in idx_sw:
                ts = df_win["timestamp"].iloc[idx]
                price = float(highs[idx])
                dist = abs(price - sr_level) / sr_level
                swings_all.append((ts, price, dist))

            swings_near = [(ts, p, d) for (ts, p, d) in swings_all if d <= tol * 2.0]
            if len(swings_near) < min_swings:
                return []

            swings_near.sort(key=lambda x: x[0])

            chosen_core = None
            chosen_close = None
            chosen_close_dist = None

            for i in range(0, len(swings_near) - min_swings + 1):
                candidate = swings_near[i : i + min_swings]
                prices = [p for (_, p, _) in candidate]

                # Cuña alcista: máximos crecientes (permitimos leve ruido)
                asc_ok = all(prices[j] >= prices[j - 1] * 0.995 for j in range(1, len(prices)))
                if not asc_ok:
                    continue

                # Duración mínima
                first_ts = candidate[0][0]
                last_ts = candidate[-1][0]
                duration_h = (last_ts - first_ts).total_seconds() / 3600.0
                if duration_h < min_hours:
                    if DEBUG_PULLBACK:
                        print(
                            f"[PULLBACK DEBUG] SHORT candidato muy corto: "
                            f"{duration_h:.1f}h < {min_hours:.1f}h → descarto"
                        )
                    continue

                # Último swing y cierre cerca del SR
                last_ts, last_price, last_dist = candidate[-1]
                pb_row = df_win[df_win["timestamp"] == last_ts]
                if pb_row.empty:
                    continue

                pb_close = float(pb_row["close"].iloc[0])
                dist_close = abs(pb_close - sr_level) / sr_level

                if max(last_dist, dist_close) > close_tol:
                    continue

                chosen_core = candidate
                chosen_close = pb_close
                chosen_close_dist = dist_close
                break

            if chosen_core is None:
                return []

            swings_core = chosen_core
            pb_close = chosen_close
            dist_close = chosen_close_dist

            swings_core_ts_prices: List[Tuple[pd.Timestamp, float]] = [
                (ts, price) for (ts, price, _) in swings_core
            ]

            if DEBUG_PULLBACK:
                _debug_swings("SHORT swings_core", swings_core_ts_prices, sr_level)

            last_ts, _, _ = swings_core[-1]
            pb_end_ts = last_ts

            pb = Pullback(
                direction=direction,
                impulse=impulse,
                swings=swings_core_ts_prices,
                end_ts=pb_end_ts,
                close=pb_close,
                dist_to_sr_pct=dist_close,
            )
            return [pb]

        # --------------------------------------------------------------
        # LONG → pullback bajista contra soporte (swing lows)
        # --------------------------------------------------------------
        elif direction == "long":
            lows = df_win["low"].to_numpy()
            idx_sw = find_swing_lows(lows)

            if len(idx_sw) < min_swings:
                return []

            swings_all: List[Tuple[pd.Timestamp, float, float]] = []
            for idx in idx_sw:
                ts = df_win["timestamp"].iloc[idx]
                price = float(lows[idx])
                dist = abs(price - sr_level) / sr_level
                swings_all.append((ts, price, dist))

            swings_near = [(ts, p, d) for (ts, p, d) in swings_all if d <= tol * 2.0]
            if len(swings_near) < min_swings:
                return []

            swings_near.sort(key=lambda x: x[0])

            chosen_core = None
            chosen_close = None
            chosen_close_dist = None

            for i in range(0, len(swings_near) - min_swings + 1):
                candidate = swings_near[i : i + min_swings]
                prices = [p for (_, p, _) in candidate]

                # Cuña bajista: mínimos decrecientes (permitimos leve ruido)
                desc_ok = all(prices[j] <= prices[j - 1] * 1.005 for j in range(1, len(prices)))
                if not desc_ok:
                    continue

                # Duración mínima
                first_ts = candidate[0][0]
                last_ts = candidate[-1][0]
                duration_h = (last_ts - first_ts).total_seconds() / 3600.0
                if duration_h < min_hours:
                    if DEBUG_PULLBACK:
                        print(
                            f"[PULLBACK DEBUG] LONG candidato muy corto: "
                            f"{duration_h:.1f}h < {min_hours:.1f}h → descarto"
                        )
                    continue

                last_ts, last_price, last_dist = candidate[-1]
                pb_row = df_win[df_win["timestamp"] == last_ts]
                if pb_row.empty:
                    continue

                pb_close = float(pb_row["close"].iloc[0])
                dist_close = abs(pb_close - sr_level) / sr_level

                if max(last_dist, dist_close) > close_tol:
                    continue

                chosen_core = candidate
                chosen_close = pb_close
                chosen_close_dist = dist_close
                break

            if chosen_core is None:
                return []

            swings_core = chosen_core
            pb_close = chosen_close
            dist_close = chosen_close_dist

            swings_core_ts_prices: List[Tuple[pd.Timestamp, float]] = [
                (ts, price) for (ts, price, _) in swings_core
            ]

            if DEBUG_PULLBACK:
                _debug_swings("LONG swings_core", swings_core_ts_prices, sr_level)

            last_ts, _, _ = swings_core[-1]
            pb_end_ts = last_ts

            pb = Pullback(
                direction=direction,
                impulse=impulse,
                swings=swings_core_ts_prices,
                end_ts=pb_end_ts,
                close=pb_close,
                dist_to_sr_pct=dist_close,
            )
            return [pb]

        # Dirección rara → sin pullbacks
        return []