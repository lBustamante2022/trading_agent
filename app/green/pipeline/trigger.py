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
        buscamos la PRIMER vela cuyo close
          > pb.close + sr_level * tol

    SHORT (impulso bajista desde resistencia):
        buscamos la PRIMER vela cuyo close
          < pb.close - sr_level * tol

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


def _log_with_ts(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = " ".join(str(a) for a in args)
    _builtins.print(f"[{ts}] {msg}", **kwargs)


print = _log_with_ts  # solo en este archivo


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

    No asume timeframes fijos.
    """

    def detect_triggers(
        self,
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
        pb_end_ts = pullback.end_ts
        pb_close = float(pullback.close)

        tol = float(getattr(cfg, "trigger_sr_tolerance", 0.05))
        max_minutes = int(getattr(cfg, "trigger_max_break_minutes", 36 * 60))

        start_ts = pb_end_ts
        end_ts = pb_end_ts + timedelta(minutes=max_minutes)

        df_win = df[
            (df["timestamp"] >= start_ts) &
            (df["timestamp"] <= end_ts)
        ].copy()

        if df_win.empty:
            if DEBUG_TRIGGER:
                print(
                    f"[TRIGGER DEBUG] ventana vacía ({start_ts} → {end_ts}) "
                    f"para TF {tf}"
                )
            return []

        if DEBUG_TRIGGER:
            print(
                f"[TRIGGER DEBUG] dir={direction}, tf={tf}, "
                f"pb_end={pb_end_ts}, sr_level={sr_level:.2f}, "
                f"pb_close={pb_close:.2f}, tol={tol:.3f}, "
                f"max_minutes={max_minutes}"
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

                if close > threshold:
                    if DEBUG_TRIGGER:
                        print(
                            f"[TRIGGER DEBUG] LONG break: ts={ts}, "
                            f"close={close:.2f} > thr={threshold:.2f}"
                        )

                    tr = Trigger(
                        direction="long",
                        timestamp=ts,
                        pullback=pullback,
                        sr_level=sr_level,
                        ref_price=pb_close,
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

                if close < threshold:
                    if DEBUG_TRIGGER:
                        print(
                            f"[TRIGGER DEBUG] SHORT break: ts={ts}, "
                            f"close={close:.2f} < thr={threshold:.2f}"
                        )

                    tr = Trigger(
                        direction="short",
                        timestamp=ts,
                        pullback=pullback,
                        sr_level=sr_level,
                        ref_price=pb_close,
                    )
                    triggers.append(tr)
                    break  # sólo el primero

        # En esta versión devolvemos 0 o 1 trigger por pullback.
        return triggers