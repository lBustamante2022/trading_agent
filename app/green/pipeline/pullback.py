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
            con el propio impulso (velas de cuerpo más chico)
          * y que, tras tocar la zona SR, aparezca una vela FUERTE
            en sentido bajista (rebote).

LONG (impulso alcista desde soporte):
    - En el TF de pullback buscamos una serie de swings bajistas que:
          * terminen cerca del SR del impulso
          * cumplan tiempos mínimo y máximo
          * movimiento lento
          * y tras tocar la zona, una vela FUERTE alcista de rebote.

Parámetros clave (GreenStyle):
    - pullback_sr_tolerance      (float)  → tolerancia al SR
    - pullback_min_hours         (float)  → duración mínima (desde fin del impulso)
    - pullback_max_days          (float)  → ventana máxima
    - (opcional) pullback_slow_body_factor (float)
          factor de comparación entre tamaño medio de cuerpo del pullback
          y el del impulso. Ej: 0.6 = pullback con cuerpos ~40% más chicos.
    - (opcional) pullback_rebound_body_min_ratio (float, default 0.6)
          porcentaje mínimo del cuerpo respecto al rango de la vela
          para considerar la vela de rebote como "fuerte".
    - (opcional) pullback_break_tol (float, default = pullback_sr_tolerance)
          tolerancia extra para invalidar si el precio rompe el SR en contra
          mientras esperamos la vela de rebote.

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
# Helper: buscar vela fuerte de rebote después de tocar SR
# ----------------------------------------------------------------------

def _find_rebound_candle(
    df_win: pd.DataFrame,
    from_ts: pd.Timestamp,
    direction: str,
    sr_level: float,
    break_tol: float,
    body_min_ratio: float,
) -> Tuple[pd.Timestamp, float, float] | None:
    """
    Busca, a partir de from_ts (excluyente), la primera vela FUERTE
    en sentido contrario al pullback (mismo sentido que el impulso).

    direction = "short" → rebote bajista (close < open)
    direction = "long"  → rebote alcista (close > open)

    Condiciones rebote:
        - body_ratio = |close-open| / (high-low) >= body_min_ratio
        - dirección correcta de la vela

    Invalida si:
        - para SHORT: close rompe resistencia demasiado por encima:
              (close - sr_level) / sr_level > break_tol
        - para LONG: close rompe soporte demasiado por debajo:
              (sr_level - close) / sr_level > break_tol
    """
    if df_win is None or df_win.empty:
        return None

    df_win = df_win.sort_values("timestamp").reset_index(drop=True)

    # Asegurar tipo Timestamp
    if not isinstance(from_ts, pd.Timestamp):
        from_ts = pd.to_datetime(from_ts)

    start_mask = df_win["timestamp"] > from_ts
    df_after = df_win[start_mask].copy()
    if df_after.empty:
        return None

    for _, row in df_after.iterrows():
        ts = row["timestamp"]
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])

        # Invalidation mientras esperamos rebote
        if direction == "short":
            # si rompe demasiado la resistencia por arriba, chau setup
            dist_break = (c - sr_level) / sr_level
            if dist_break > break_tol:
                print(
                    f"[PULLBACK SHORT] invalidado esperando rebote: "
                    f"{ts} close={c:.2f}, dist_break={dist_break:.4f} > break_tol={break_tol:.4f}"
                )
                return None
        else:  # long
            dist_break = (sr_level - c) / sr_level
            if dist_break > break_tol:
                print(
                    f"[PULLBACK LONG] invalidado esperando rebote: "
                    f"{ts} close={c:.2f}, dist_break={dist_break:.4f} > break_tol={break_tol:.4f}"
                )
                return None

        # cuerpo de la vela
        rang = max(h - l, 1e-9)
        body = abs(c - o)
        body_ratio = body / rang

        cond_body = body_ratio >= body_min_ratio
        if direction == "short":
            cond_dir = c < o  # vela bajista
        else:
            cond_dir = c > o  # vela alcista

        if cond_dir and cond_body:
            print(
                f"[PULLBACK {direction.upper()}] rebote fuerte @ {ts} "
                f"close={c:.2f} body_ratio={body_ratio:.3f}"
            )
            return ts, c, body_ratio

    # No se encontró vela fuerte
    return None


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
        pullback_slow_body_factor          (float, default 0.6)
        pullback_rebound_body_min_ratio    (float, default 0.6)
        pullback_break_tol                 (float, default pullback_sr_tolerance)

    Idea:
        - El pullback debe ser "lento" vs el impulso:
            avg_body_pct_pullback <= pullback_slow_body_factor * avg_body_pct_impulse

        - El "FIN" real del pullback ocurre cuando:
            * el precio ya llegó a la zona SR (swings cerca del nivel), y
            * aparece una vela FUERTE en sentido del impulso
              (rebote claro desde el SR).
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
        start_ts = impulse.meta["end"]
        end_ts = impulse.meta["end"] + timedelta(days=max_days)

        df_win = df[
            (df["timestamp"] > start_ts) &
            (df["timestamp"] <= end_ts)
        ].copy()

        if df_win.empty:
            print(f"[{symbol}] ventana vacía pullback ({start_ts} → {end_ts})")
            return []

        sr_level = float(impulse.meta["sr_level"])
        direction = impulse.meta["direction"]
        sr_kind = impulse.meta.get("sr_kind", "support" if direction == "long" else "resistance")

        # Parámetros de estilo
        tol_sr = float(style.pullback_sr_tolerance)
        min_hours = float(style.pullback_min_hours)
        slow_factor = float(getattr(style, "pullback_slow_body_factor", 0.6))
        rebound_body_min_ratio = float(getattr(style, "pullback_rebound_body_min_ratio", 0.6))
        break_tol = float(getattr(style, "pullback_break_tol", tol_sr))

        # --------------------------------------------------------------
        # Cálculo de "velocidad" del impulso (tamaño medio de velas)
        # --------------------------------------------------------------
        impulse_tf = style.impulse_tf
        df_impulse_tf = dfs.get(impulse_tf)
        avg_body_impulse_pct: float | None = None

        if df_impulse_tf is not None and not df_impulse_tf.empty:
            df_imp_seg = df_impulse_tf[
                (df_impulse_tf["timestamp"] >= impulse.meta["start"]) &
                (df_impulse_tf["timestamp"] <= impulse.meta["end"])
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
            f"sr_level={sr_level:.2f}, avg_body_impulse={avg_body_impulse_pct:.4f}, "
            f"rebound_body_min_ratio={rebound_body_min_ratio:.2f}, break_tol={break_tol:.4f}"
        )

        # --------------------------------------------------------------
        # SHORT → pullback alcista contra resistencia (swing highs)
        # --------------------------------------------------------------
        if direction == "short":
            highs = df_win["high"].to_numpy()
            idx_sw = find_swing_highs(highs)

            swings_all: List[Tuple[pd.Timestamp, float, float]] = []
            for idx in idx_sw:
                ts_sw = df_win["timestamp"].iloc[idx]
                price = float(highs[idx])
                dist_sr = abs(price - sr_level) / sr_level
                swings_all.append((ts_sw, price, dist_sr))

            for i in range(len(swings_all)):
                last_ts, last_price, last_dist = swings_all[i]

                # 1) ¿Llegamos a la zona SR?
                if last_dist > tol_sr:
                    continue

                candidate_swings = swings_all[: i + 1]
                swings_core_ts_prices: List[Tuple[pd.Timestamp, float]] = [
                    (ts_c, price_c) for (ts_c, price_c, _) in candidate_swings
                ]

                # 2) Duración mínima desde fin del impulso
                duration_h = (last_ts - impulse.meta["end"]).total_seconds() / 3600.0
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
                pb_close_sr_touch = float(pb_row["close"].iloc[0])
                dist_close = abs(pb_close_sr_touch - sr_level) / sr_level
                if dist_close > tol_sr:
                    continue

                # 4) LENTEZA del pullback: cuerpos más chicos que el impulso
                df_pb_seg = df_win[
                    (df_win["timestamp"] > impulse.meta["end"]) &
                    (df_win["timestamp"] <= last_ts)
                ].copy()
                avg_body_pb_pct = _avg_body_pct(df_pb_seg)

                if avg_body_pb_pct is None:
                    print(f"[{symbol} SHORT] no pude medir cuerpo del pullback → descarto")
                    continue

                max_allowed_pb = slow_factor * avg_body_impulse_pct
                if avg_body_pb_pct > max_allowed_pb:
                    print(
                        f"[{symbol} SHORT] pullback demasiado rápido: "
                        f"avg_body_pb={avg_body_pb_pct:.4f} > "
                        f"slow_factor * avg_body_impulse={max_allowed_pb:.4f} → descarto"
                    )
                    continue

                # 5) BUSCAR VELA FUERTE DE REBOTE BAJISTA DESPUÉS DEL TOQUE
                rebound = _find_rebound_candle(
                    df_win=df_win,
                    from_ts=last_ts,
                    direction=direction,
                    sr_level=sr_level,
                    break_tol=break_tol,
                    body_min_ratio=rebound_body_min_ratio,
                )
                if rebound is None:
                    # No hubo rebote "claro" después de tocar la resistencia
                    continue

                rebound_ts, rebound_close, rebound_body_ratio = rebound

                _debug_swings(f"{symbol} SHORT swings_core", swings_core_ts_prices, sr_level)

                pb = Pullback(
                    impulse=impulse,
                    meta={
                        "symbol": symbol,
                        "direction": direction,
                        "start": start_ts,
                        "end": rebound_ts,          # FIN REAL DEL PULLBACK = vela de rebote
                        "end_close": rebound_close, # close de la vela de rebote
                        "sr_level": sr_level,
                        "sr_kind": sr_kind,
                        "sr_touch_ts": last_ts,
                        "sr_touch_price": last_price,
                        "sr_touch_dist_pct": last_dist,
                        "avg_body_impulse_pct": avg_body_impulse_pct,
                        "avg_body_pullback_pct": avg_body_pb_pct,
                        "rebound_body_ratio": rebound_body_ratio,
                        "rebound_ts": rebound_ts,
                        "rebound_dir": "down",
                    },
                )
                return [pb]

            # Ningún candidato válido
            return []

        # --------------------------------------------------------------
        # LONG → pullback bajista contra soporte (swing lows)
        # --------------------------------------------------------------
        elif direction == "long":
            lows = df_win["low"].to_numpy()
            idx_sw = find_swing_lows(lows)

            swings_all: List[Tuple[pd.Timestamp, float, float]] = []
            for idx in idx_sw:
                ts_sw = df_win["timestamp"].iloc[idx]
                price = float(lows[idx])
                dist_sr = abs(price - sr_level) / sr_level
                swings_all.append((ts_sw, price, dist_sr))

            for i in range(len(swings_all)):
                last_ts, last_price, last_dist = swings_all[i]

                # 1) Llegada a zona SR
                if last_dist > tol_sr:
                    continue

                candidate_swings = swings_all[: i + 1]
                swings_core_ts_prices: List[Tuple[pd.Timestamp, float]] = [
                    (ts_c, price_c) for (ts_c, price_c, _) in candidate_swings
                ]

                # 2) Duración mínima desde fin del impulso
                duration_h = (last_ts - impulse.meta["end"]).total_seconds() / 3600.0
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
                pb_close_sr_touch = float(pb_row["close"].iloc[0])
                dist_close = abs(pb_close_sr_touch - sr_level) / sr_level
                if dist_close > tol_sr:
                    continue

                # 4) LENTEZA del pullback bajista
                df_pb_seg = df_win[
                    (df_win["timestamp"] > impulse.meta["end"]) &
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

                # 5) BUSCAR VELA FUERTE DE REBOTE ALCISTA DESPUÉS DEL TOQUE
                rebound = _find_rebound_candle(
                    df_win=df_win,
                    from_ts=last_ts,
                    direction=direction,
                    sr_level=sr_level,
                    break_tol=break_tol,
                    body_min_ratio=rebound_body_min_ratio,
                )
                if rebound is None:
                    continue

                rebound_ts, rebound_close, rebound_body_ratio = rebound

                _debug_swings(f"{symbol} LONG swings_core", swings_core_ts_prices, sr_level)

                pb = Pullback(
                    impulse=impulse,
                    meta={
                        "symbol": symbol,
                        "direction": direction,
                        "start": start_ts,
                        "end": rebound_ts,
                        "end_close": rebound_close,
                        "sr_level": sr_level,
                        "sr_kind": sr_kind,
                        "sr_touch_ts": last_ts,
                        "sr_touch_price": last_price,
                        "sr_touch_dist_pct": last_dist,
                        "avg_body_impulse_pct": avg_body_impulse_pct,
                        "avg_body_pullback_pct": avg_body_pb_pct,
                        "rebound_body_ratio": rebound_body_ratio,
                        "rebound_ts": rebound_ts,
                        "rebound_dir": "up",
                    },
                )
                return [pb]

            return []

        # Dirección rara → sin pullbacks
        return []