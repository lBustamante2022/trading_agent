# app/green/pipeline/pullback.py
"""
Detección de PULLBACK para GREEN v3 (arquitectura nueva).

SHORT (impulso bajista desde resistencia):
    - En el TF de pullback (style.pullback_tf) buscamos una serie
      de swings alcistas posteriores al impulso que:
          * terminen cerca del nivel SR del impulso
          * tarden al menos pullback_min_hours desde el FIN del impulso
          * estén dentro de una ventana de máximo pullback_max_days
          * se muevan de forma LENTA / NO ACELERADA en comparación
            con el propio impulso (velas de cuerpo más chico).

LONG (impulso alcista desde soporte):
    - En el TF de pullback buscamos una serie de swings bajistas que:
          * terminen cerca del SR del impulso
          * cumplan tiempos mínimo y máximo
          * y cuyo movimiento también sea lento vs el impulso.

Parámetros clave (GreenStyle):
    - pullback_sr_tolerance      (float)  → tolerancia al SR
    - pullback_min_hours         (float)  → duración mínima (desde fin del impulso)
    - pullback_max_days          (float)  → ventana máxima
    - (opcional) pullback_slow_body_factor (float)
          factor de comparación entre tamaño medio de cuerpo del pullback
          y el del impulso. Ej: 0.6 = pullback con cuerpos ~40% más chicos.

No asume timeframes fijos, toma el TF de style.pullback_tf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

from datetime import datetime, timedelta

import builtins as _builtins
import numpy as np
import pandas as pd

from app.green.core import PullbackStrategy, Pullback, Impulse
from app.green.styles import GreenStyle
from app.ta.swing_points import find_swing_highs, find_swing_lows


# ----------------------------------------------------------------------
# DEBUG PRINT LOCAL (opcional)
# ----------------------------------------------------------------------

DEBUG_PULLBACK = False


def _log(*args, **kwargs):
    if DEBUG_PULLBACK:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = " ".join(str(a) for a in args)
        _builtins.print(f"[{ts} DEBUG PULLBACK] {msg}", **kwargs)


print = _log  # override local


def _debug_swings(label: str, swings: List[Tuple[pd.Timestamp, float]], sr: float):
    if DEBUG_PULLBACK:
        print(f"{label}: {len(swings)} swings")
        for ts, price in swings:
            dist = abs(price - sr) / sr
            print(f"   - {ts}  price={price:.2f}  dist_sr={dist:.4f}")


# ----------------------------------------------------------------------
# Helper: tamaño medio de cuerpo en % (|close-open| / |close|)
# ----------------------------------------------------------------------

def _avg_body_pct(df: pd.DataFrame) -> float | None:
    """
    Devuelve el tamaño medio del cuerpo de las velas en términos relativos:

        body_pct = |close - open| / |close|

    Ignora filas con close=0 o NaN. Si no hay datos válidos, devuelve None.
    """
    if df is None or df.empty:
        return None
    if "open" not in df.columns or "close" not in df.columns:
        return None

    closes = df["close"].astype(float)
    opens = df["open"].astype(float)

    body = (closes - opens).abs()
    denom = closes.abs().replace(0, np.nan)

    pct = (body / denom).replace([np.inf, -np.inf], np.nan).dropna()
    if pct.empty:
        return None
    return float(pct.mean())


# ----------------------------------------------------------------------
# Estrategia concreta de Pullback
# ----------------------------------------------------------------------

@dataclass
class DefaultPullbackStrategy(PullbackStrategy):
    """
    Estrategia genérica de pullback contra SR, sin validar cuña geométrica.

    Parametría tomada del estilo (GreenStyle):

        pullback_sr_tolerance      (float)
        pullback_min_hours         (float)
        pullback_max_days          (float)

    Adicional (opcional, también en GreenStyle si querés):
        pullback_slow_body_factor  (float, default 0.6)

    Idea:
        - El pullback debe ser "lento" vs el impulso:
            avg_body_pct_pullback <= pullback_slow_body_factor * avg_body_pct_impulse
    """

    def detect_pullbacks(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,          # se mantiene en la firma por compatibilidad
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
        # Ventana temporal después del impulso: desde FIN del impulso
        # --------------------------------------------------------------
        max_days = float(style.pullback_max_days)
        start_ts = impulse.end
        end_ts = impulse.end + timedelta(days=max_days)

        df_win = df[
            (df["timestamp"] > start_ts) &
            (df["timestamp"] <= end_ts)
        ].copy()

        if df_win.empty:
            print(f"[{symbol}] ventana vacía pullback ({start_ts} → {end_ts})")
            return []

        sr_level = float(impulse.sr_level)
        direction = impulse.direction

        # Parámetros de estilo
        tol_sr = float(style.pullback_sr_tolerance)
        min_hours = float(style.pullback_min_hours)
        slow_factor = float(getattr(style, "pullback_slow_body_factor", 0.6))

        # --------------------------------------------------------------
        # Cálculo de "velocidad" del impulso (tamaño medio de velas)
        # --------------------------------------------------------------
        impulse_tf = style.impulse_tf
        df_impulse_tf = dfs.get(impulse_tf)
        avg_body_impulse_pct: float | None = None

        if df_impulse_tf is not None and not df_impulse_tf.empty:
            df_imp_seg = df_impulse_tf[
                (df_impulse_tf["timestamp"] >= impulse.start) &
                (df_impulse_tf["timestamp"] <= impulse.end)
            ].copy()
            avg_body_impulse_pct = _avg_body_pct(df_imp_seg)

        # Fallback: si no pudimos medir el impulso, usamos el parámetro
        # impulse_min_body_pct como referencia para "velas grandes".
        if avg_body_impulse_pct is None:
            avg_body_impulse_pct = float(
                getattr(style, "impulse_min_body_pct", 0.02)
            )

        print(
            f"[{symbol} PULLBACK] dir={direction}, tf={tf}, "
            f"max_days={max_days}, sr_tol={tol_sr:.4f}, "
            f"min_hours={min_hours:.1f}, slow_factor={slow_factor:.2f}, "
            f"sr_level={sr_level:.2f}, avg_body_impulse={avg_body_impulse_pct:.4f}"
        )

        # --------------------------------------------------------------
        # SHORT → pullback alcista contra resistencia (swing highs)
        # --------------------------------------------------------------
        if direction == "short":
            highs = df_win["high"].to_numpy()
            idx_sw = find_swing_highs(
                highs,
                # left=2,
                # right=2,
                # min_prominence_pct=0.004,
                # min_separation=10,
            )
            # idx_sw = []
            # for i in range(1, len(highs) - 1):
            #     idx_sw.append(i)

            # Swings con distancia al SR
            swings_all: List[Tuple[pd.Timestamp, float, float]] = []
            for idx in idx_sw:
                ts = df_win["timestamp"].iloc[idx]
                price = float(highs[idx])
                dist_sr = abs(price - sr_level) / sr_level
                swings_all.append((ts, price, dist_sr))

            # Recorremos swings en orden; cuando un swing entra en SR,
            # validamos: tiempo, cercanía y "lenteza" del movimiento.
            for i in range(len(swings_all)):
                last_ts, last_price, last_dist = swings_all[i]

                # 1) ¿Llegamos a la zona SR?
                if last_dist > tol_sr:
                    continue

                # Swings usados: desde el primero hasta este
                candidate_swings = swings_all[: i + 1]
                swings_core_ts_prices: List[Tuple[pd.Timestamp, float]] = [
                    (ts, price) for (ts, price, _) in candidate_swings
                ]

                # 2) Duración mínima desde fin del impulso
                duration_h = (last_ts - impulse.end).total_seconds() / 3600.0
                if duration_h < min_hours:
                    print(
                        f"[{symbol} SHORT] candidato muy corto (desde fin impulso): "
                        f"{duration_h:.1f}h < {min_hours:.1f}h → descarto"
                    )
                    continue

                # 3) Cierre cerca del SR en el último swing
                pb_row = df_win[df_win["timestamp"] == last_ts]
                if pb_row.empty:
                    continue
                pb_close = float(pb_row["close"].iloc[0])
                dist_close = abs(pb_close - sr_level) / sr_level
                if dist_close > tol_sr:
                    continue

                # 4) LENTEZA del pullback: cuerpos más chicos que el impulso
                df_pb_seg = df_win[
                    (df_win["timestamp"] > impulse.end) &
                    (df_win["timestamp"] <= last_ts)
                ].copy()
                avg_body_pb_pct = _avg_body_pct(df_pb_seg)

                if avg_body_pb_pct is None:
                    # si no podemos medir, por seguridad descartamos el candidato
                    print(f"[{symbol} SHORT] no pude medir cuerpo del pullback → descarto")
                    continue

                # condición de "pullback lento"
                max_allowed_pb = slow_factor * avg_body_impulse_pct
                if avg_body_pb_pct > max_allowed_pb:
                    print(
                        f"[{symbol} SHORT] pullback demasiado rápido: "
                        f"avg_body_pb={avg_body_pb_pct:.4f} > "
                        f"slow_factor * avg_body_impulse={max_allowed_pb:.4f} → descarto"
                    )
                    continue

                # OK → Pullback válido
                _debug_swings(f"{symbol} SHORT swings_core", swings_core_ts_prices, sr_level)

                pb = Pullback(
                    symbol=symbol,
                    direction=direction,
                    impulse=impulse,
                    # swings=swings_core_ts_prices,
                    start=start_ts,
                    end=last_ts,
                    end_close=pb_close
                )
                return [pb]

            # Ningún candidato válido
            return []

        # --------------------------------------------------------------
        # LONG → pullback bajista contra soporte (swing lows)
        # --------------------------------------------------------------
        elif direction == "long":
            lows = df_win["low"].to_numpy()
            idx_sw = find_swing_lows(
                lows,
                # left=2,
                # right=2,
                # min_prominence_pct=0.004,
                # min_separation=10,
            )
            # idx_sw = []
            # for i in range(1, len(lows) - 1):
            #     idx_sw.append(i)

            swings_all: List[Tuple[pd.Timestamp, float, float]] = []
            for idx in idx_sw:
                ts = df_win["timestamp"].iloc[idx]
                price = float(lows[idx])
                dist_sr = abs(price - sr_level) / sr_level
                swings_all.append((ts, price, dist_sr))

            for i in range(len(swings_all)):
                last_ts, last_price, last_dist = swings_all[i]

                # 1) Llegada a zona SR
                if last_dist > tol_sr:
                    continue

                candidate_swings = swings_all[: i + 1]
                swings_core_ts_prices: List[Tuple[pd.Timestamp, float]] = [
                    (ts, price) for (ts, price, _) in candidate_swings
                ]

                # 2) Duración mínima desde fin del impulso
                duration_h = (last_ts - impulse.end).total_seconds() / 3600.0
                if duration_h < min_hours:
                    print(
                        f"[{symbol} LONG] candidato muy corto (desde fin impulso): "
                        f"{duration_h:.1f}h < {min_hours:.1f}h → descarto"
                    )
                    continue

                # 3) Cierre cerca de SR
                pb_row = df_win[df_win["timestamp"] == last_ts]
                if pb_row.empty:
                    continue
                pb_close = float(pb_row["close"].iloc[0])
                dist_close = abs(pb_close - sr_level) / sr_level
                if dist_close > tol_sr:
                    continue

                # 4) LENTEZA del pullback bajista
                df_pb_seg = df_win[
                    (df_win["timestamp"] > impulse.end) &
                    (df_win["timestamp"] <= last_ts)
                ].copy()
                avg_body_pb_pct = _avg_body_pct(df_pb_seg)

                if avg_body_pb_pct is None:
                    print(f"[{symbol} LONG] no pude medir cuerpo del pullback → descarto")
                    continue

                max_allowed_pb = slow_factor * avg_body_impulse_pct
                if avg_body_pb_pct > max_allowed_pb:
                    print(
                        f"[{symbol} LONG] pullback demasiado rápido: "
                        f"avg_body_pb={avg_body_pb_pct:.4f} > "
                        f"slow_factor * avg_body_impulse={max_allowed_pb:.4f} → descarto"
                    )
                    continue

                _debug_swings(f"{symbol} LONG swings_core", swings_core_ts_prices, sr_level)

                pb = Pullback(
                    symbol=symbol,
                    direction=direction,
                    impulse=impulse,
                    # swings=swings_core_ts_prices,
                    start=start_ts,
                    end=last_ts,
                    end_close=pb_close
                )
                return [pb]

            return []

        # Dirección rara → sin pullbacks
        return []