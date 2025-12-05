# app/green/pipeline/impulse.py
"""
Detección de IMPULSOS para GREEN v3 (arquitectura nueva).

- Usa swings para detectar niveles SR horizontales.
- Para cada SR, detecta impulsos cuando el precio rompe con fuerza
  ese nivel (breakout) y el cuerpo de la vela supera un umbral
  mínimo (impulse_min_body_pct del estilo).

No asume timeframes fijos: toma el TF desde GreenStyle.impulse_tf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Literal

import numpy as np
import pandas as pd

from app.green.core import ImpulseStrategy, Impulse
from app.green.styles import GreenStyle
from app.ta.swing_points import find_swing_highs, find_swing_lows


# ----------------------------------------------------------------------
# Estructura interna de nivel SR
# ----------------------------------------------------------------------

@dataclass
class SwingSRLevel:
    level: float
    kind: Literal["support", "resistance"]
    touches: int
    first_ts: pd.Timestamp
    last_ts: pd.Timestamp


# ----------------------------------------------------------------------
# Utilidades para agrupar swings en niveles SR
# ----------------------------------------------------------------------

def _cluster_swings(
    prices: np.ndarray,
    idxs: np.ndarray,
    timestamps: pd.Series,
    kind: Literal["support", "resistance"],
    price_tol_pct: float,
    min_touches: int,
) -> List[SwingSRLevel]:
    """
    Agrupa swings cercanos en precio dentro de `price_tol_pct`
    y devuelve niveles SR representativos.
    """
    clusters: List[Dict[str, Any]] = []

    for i in idxs:
        p = float(prices[i])
        ts = timestamps.iloc[i]

        found = False
        for cl in clusters:
            lvl = cl["level"]
            if abs(p - lvl) / lvl <= price_tol_pct:
                cl["prices"].append(p)
                cl["touches"] += 1
                cl["first_ts"] = min(cl["first_ts"], ts)
                cl["last_ts"] = max(cl["last_ts"], ts)
                found = True
                break

        if not found:
            clusters.append(
                {
                    "level": p,
                    "prices": [p],
                    "touches": 1,
                    "first_ts": ts,
                    "last_ts": ts,
                }
            )

    levels: List[SwingSRLevel] = []
    for cl in clusters:
        if cl["touches"] < min_touches:
            continue
        lvl = float(np.mean(cl["prices"]))
        levels.append(
            SwingSRLevel(
                level=lvl,
                kind=kind,
                touches=cl["touches"],
                first_ts=cl["first_ts"],
                last_ts=cl["last_ts"],
            )
        )
    return levels


def _detect_swing_sr_levels(
    df: pd.DataFrame,
    price_tol_pct: float = 0.005,  # ~0.5%
    min_touches: int = 5,
) -> List[SwingSRLevel]:
    """
    Detecta niveles SR horizontales a partir de swings de highs y lows
    en el TF de impulso.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    ts = df["timestamp"]

    idx_highs = np.array(find_swing_highs(highs), dtype=int)
    idx_lows = np.array(find_swing_lows(lows), dtype=int)

    res_levels = _cluster_swings(
        prices=highs,
        idxs=idx_highs,
        timestamps=ts,
        kind="resistance",
        price_tol_pct=price_tol_pct,
        min_touches=min_touches,
    )

    sup_levels = _cluster_swings(
        prices=lows,
        idxs=idx_lows,
        timestamps=ts,
        kind="support",
        price_tol_pct=price_tol_pct,
        min_touches=min_touches,
    )

    return res_levels + sup_levels


# ----------------------------------------------------------------------
# Estrategia concreta de impulsos
# ----------------------------------------------------------------------

@dataclass
class DefaultImpulseStrategy(ImpulseStrategy):
    """
    Estrategia de impulso genérica para GREEN v3.

    Parámetros usados:

      - style.impulse_tf           → TF a usar.
      - style.impulse_min_body_pct → tamaño mínimo del cuerpo de la vela
                                     para considerar un impulso.
      - cfg.channel_width_pct      → se reutiliza como tolerancia para
                                     agrupar SR (en %).
      - cfg.impulse_break_pct (opcional)
        → % mínimo de ruptura del nivel SR (default: 1%).

    Un "impulso" se produce cuando:

      - Para resistencia (SHORT):
          * vela anterior cierra bajo el SR,
          * vela actual cierra claramente por debajo (ruptura),
          * cuerpo de la vela (|close-open|/open) >= impulse_min_body_pct.

      - Para soporte (LONG): análogo pero al alza.
    """

    def detect_impulses(
        self,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
    ) -> List[Impulse]:

        tf = style.impulse_tf
        df = dfs.get(tf)
        if df is None or df.empty:
            return []

        if "timestamp" not in df.columns or "open" not in df.columns or "close" not in df.columns:
            return []

        df = df.sort_values("timestamp").reset_index(drop=True)

        # --- 1) Detectar niveles SR horizontales por swings ---
        sr_tol_pct = float(getattr(cfg, "channel_width_pct", 5.0)) / 100.0
        sr_levels = _detect_swing_sr_levels(
            df,
            price_tol_pct=sr_tol_pct,
            min_touches=5,
        )
        if not sr_levels:
            return []

        closes = df["close"].to_numpy()
        opens = df["open"].to_numpy()
        ts = df["timestamp"]

        break_pct = float(getattr(cfg, "impulse_break_pct", 0.01))  # 1% por defecto
        min_body_pct = float(getattr(style, "impulse_min_body_pct", 0.02))

        impulses: List[Impulse] = []

        n = len(df)
        for lvl in sr_levels:
            level = float(lvl.level)

            i = 1
            while i < n:
                c_prev = float(closes[i - 1])
                c_now = float(closes[i])
                o_now = float(opens[i])
                t_now = ts.iloc[i]

                # tamaño del cuerpo en %
                body_pct = abs(c_now - o_now) / max(o_now, 1e-9)

                if body_pct < min_body_pct:
                    i += 1
                    continue

                if lvl.kind == "support":
                    # Ruptura bajista del soporte → impulso SHORT
                    if c_prev >= level and c_now < level * (1.0 - break_pct):
                        impulses.append(
                            Impulse(
                                direction="short",
                                start=t_now,
                                end=t_now,
                                sr_level=level,
                            )
                        )
                        # saltamos unas velas para no duplicar impulsos pegados
                        i += 3
                        continue

                else:  # resistance
                    # Ruptura alcista de la resistencia → impulso LONG
                    if c_prev <= level and c_now > level * (1.0 + break_pct):
                        impulses.append(
                            Impulse(
                                direction="long",
                                start=t_now,
                                end=t_now,
                                sr_level=level,
                            )
                        )
                        i += 3
                        continue

                i += 1

        # Ordenamos por tiempo por prolijidad
        impulses.sort(key=lambda imp: imp.start)
        return impulses