# app/green/pipeline/trigger.py
"""
Detección de TRIGGER para GREEN v3 (arquitectura nueva).

Responsabilidad:

    A partir de un Pullback ya válido (cuña contra SR),
    buscar la vela de ruptura que dispara el trade.

Idea simplificada y genérica:

    - Usamos el TF de trigger: style.trigger_tf
    - Ventana temporal:
          desde pullback.end_ts (inclusive)
          hasta pullback.end_ts + cfg.trigger_max_break_minutes
    - Nivel de referencia:
          precio de cierre del último swing del pullback (pb.close)
    - Tolerancia:
          cfg.trigger_sr_tolerance (porcentaje sobre sr_level)

    LONG  (impulso alcista desde soporte):
        - buscamos la PRIMER vela cuyo close
              > pb.close + sr_level * tol
        - si en el camino el precio ROMPE el soporte por debajo de
          sr_level - sr_level * break_tol  → setup INVALIDADO.

    SHORT (impulso bajista desde resistencia):
        - buscamos la PRIMER vela cuyo close
              < pb.close - sr_level * tol
        - si en el camino el precio ROMPE la resistencia por encima de
          sr_level + sr_level * break_tol → setup INVALIDADO.

Parámetros de invalidación (tomados del estilo GreenStyle):
    - pullback_min_tolerance (float)  → tolerancia extra para ruptura del SR
      (si no existe, se usa pullback_sr_tolerance como fallback).

Devuelve: lista de Trigger (en esta versión, 0 o 1 por pullback).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any

from datetime import datetime, timedelta
import builtins as _builtins

import pandas as pd

from app.green.core import TriggerStrategy, Trigger, Pullback
from app.green.styles import GreenStyle


# ----------------------------------------------------------------------
# DEBUG LOCAL
# ----------------------------------------------------------------------

DEBUG_TRIGGER = False

def _log(*args, **kwargs):
    if DEBUG_TRIGGER:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = " ".join(str(a) for a in args)
        _builtins.print(f"[{ts} DEBUG TRIGGER] {msg}", **kwargs)

print = _log  # override local


# ----------------------------------------------------------------------
# Estrategia concreta de TRIGGER
# ----------------------------------------------------------------------

@dataclass
class DefaultTriggerStrategy(TriggerStrategy):
    """
    Estrategia genérica de ruptura posterior al pullback.

    Usa:

      De cfg:
        - trigger_sr_tolerance (float)          p.ej. 0.05
        - trigger_max_break_minutes (int)       ventana máxima de espera

      Del style:
        - trigger_tf (str)
        - pullback_min_tolerance (float, opcional)
        - pullback_sr_tolerance (float, fallback si no hay min_tolerance)

    No asume timeframes fijos.
    """

    def detect_triggers(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        pullback: Pullback,
    ) -> List[Trigger]:

        tf = style.trigger_tf
        df = dfs.get(tf)
        if df is None or df.empty:
            return []

        if "timestamp" not in df.columns or "close" not in df.columns:
            return []

        df = df.sort_values("timestamp").reset_index(drop=True)

        direction = pullback.direction       # "long" / "short" (misma que el impulso)
        sr_level = float(pullback.impulse.sr_level)
        pb_end_ts = pullback.end
        pb_close = float(pullback.end_close)

        tol = float(getattr(cfg, "trigger_sr_tolerance", 0.05))
        max_minutes = int(getattr(cfg, "trigger_max_break_minutes", 36 * 60))

        # Tolerancia de INVALIDACIÓN del setup si se rompe el SR en contra
        break_tol = float(
            getattr(
                style,
                "pullback_min_tolerance",
                getattr(style, "pullback_sr_tolerance", tol),
            )
        )

        start_ts = pb_end_ts
        end_ts = pb_end_ts + timedelta(minutes=max_minutes)

        df_win = df[
            (df["timestamp"] >= start_ts) &
            (df["timestamp"] <= end_ts)
        ].copy()

        if df_win.empty:
            print(
                f"ventana vacía ({start_ts} → {end_ts}) "
                f"para TF {tf}"
            )
            return []

        print(
            f"dir={direction}, tf={tf}, "
            f"pb_end={pb_end_ts}, sr_level={sr_level:.2f}, "
            f"pb_close={pb_close:.2f}, tol={tol:.3f}, "
            f"break_tol={break_tol:.4f}, max_minutes={max_minutes}"
        )

        triggers: List[Trigger] = []

        # --------------------------------------------------------------
        # LONG: ruptura alcista tras pullback bajista a soporte
        # --------------------------------------------------------------
        if direction == "long":
            # nivel de ruptura = cierre del pullback + tolerancia sobre el SR
            threshold = pb_close + sr_level * tol

            for _, row in df_win.iterrows():
                ts = row["timestamp"]
                close = float(row["close"])

                # 1) INVALIDACIÓN: ¿rompió el SOPORTE demasiado hacia abajo?
                #    Si (sr_level - close) / sr_level > break_tol → descartar setup.
                dist_break = (sr_level - close) / sr_level
                if dist_break > break_tol:
                    print(
                        f"LONG invalidado: ts={ts}, close={close:.2f} "
                        f"rompe soporte más allá de tolerancia "
                        f"(dist_break={dist_break:.4f} > break_tol={break_tol:.4f})"
                    )
                    return []

                # 2) Trigger válido: ruptura alcista
                if close > threshold:
                    print(
                        f"LONG break: ts={ts}, "
                        f"close={close:.2f} > thr={threshold:.2f}"
                    )

                    tr = Trigger(
                        symbol=symbol,
                        direction="long",
                        start=start_ts,
                        end=ts,
                        pullback=pullback,
                        end_close=pb_close,
                    )
                    triggers.append(tr)
                    break  # sólo el primero

        # --------------------------------------------------------------
        # SHORT: ruptura bajista tras pullback alcista a resistencia
        # --------------------------------------------------------------
        elif direction == "short":
            threshold = pb_close - sr_level * tol

            for _, row in df_win.iterrows():
                ts = row["timestamp"]
                close = float(row["close"])

                # 1) INVALIDACIÓN: ¿rompió la RESISTENCIA demasiado hacia arriba?
                #    Si (close - sr_level) / sr_level > break_tol → descartar setup.
                dist_break = (close - sr_level) / sr_level
                if dist_break > break_tol:
                    print(
                        f"SHORT invalidado: ts={ts}, close={close:.2f} "
                        f"rompe resistencia más allá de tolerancia "
                        f"(dist_break={dist_break:.4f} > break_tol={break_tol:.4f})"
                    )
                    return []

                # 2) Trigger válido: ruptura bajista
                if close < threshold:
                    print(
                        f"SHORT break: ts={ts}, "
                        f"close={close:.2f} < thr={threshold:.2f}"
                    )

                    tr = Trigger(
                        symbol=symbol,
                        direction="short",
                        start=start_ts,
                        end=ts,
                        pullback=pullback,
                        end_close=pb_close,
                    )
                    triggers.append(tr)
                    break  # sólo el primero

        # En esta versión devolvemos 0 o 1 trigger por pullback.
        return triggers