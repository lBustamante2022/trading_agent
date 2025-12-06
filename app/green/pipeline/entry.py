# app/green/pipeline/entry.py
"""
Detección de ENTRY para GREEN v3 (arquitectura nueva).

Reglas:

    - TF usado: style.entry_tf
    - Ventana temporal:
          desde trigger.timestamp
          hasta trigger.timestamp + cfg.entry_lookahead_minutes
    - Confirmación mínima:
          LONG  → close > trigger.ref_price
          SHORT → close < trigger.ref_price
    - SL y TP se arman con tolerancias configurables:
          sl_buffer_pct (sobre entry_price)
          tp_rr (factor mínimo RR planificado)

Devuelve 0 o 1 Entry por trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
import builtins as _builtins

import pandas as pd

from app.green.core import Entry, Trigger
from app.green.styles import GreenStyle
from app.ai.green_supervisor import review_setup_with_ai

# ----------------------------------------------------------------------
# DEBUG PRINT LOCAL
# ----------------------------------------------------------------------

DEBUG_ENTRY = False

def _log(*args, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = " ".join(str(a) for a in args)
    _builtins.print(f"[{ts}] {msg}", **kwargs)

print = _log  # override local


# ----------------------------------------------------------------------
# ENTRY STRATEGY
# ----------------------------------------------------------------------

@dataclass
class DefaultEntryStrategy:
    """
    Estrategia genérica de entrada posterior al trigger.

    Parámetros usados desde cfg:
      - entry_lookahead_minutes
      - sl_buffer_pct
      - min_rr

    Del style:
      - entry_tf
    """

    def detect_entry(
        self,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        trigger: Trigger,
        ai_supervisor: bool = False,
    ) -> Optional[Entry]:

        tf = style.entry_tf
        df = dfs.get(tf)
        if df is None or df.empty:
            return None

        if "timestamp" not in df.columns or "close" not in df.columns:
            return None

        df = df.sort_values("timestamp").reset_index(drop=True)

        lookahead = int(getattr(cfg, "entry_lookahead_minutes", 60))
        sl_buf = float(getattr(cfg, "sl_buffer_pct", 0.002))
        min_rr = float(getattr(cfg, "min_rr", 2.0))

        direction = trigger.direction
        start_ts = trigger.timestamp
        end_ts = start_ts + timedelta(minutes=lookahead)

        ref_price = float(trigger.ref_price)
        sr_level = float(trigger.sr_level)

        df_win = df[
            (df["timestamp"] >= start_ts) &
            (df["timestamp"] <= end_ts)
        ].copy()

        if df_win.empty:
            return None

        if DEBUG_ENTRY:
            print(
                f"[ENTRY DEBUG] dir={direction}, tf={tf}, "
                f"start={start_ts}, end={end_ts}, ref_price={ref_price:.2f}"
            )

        # --------------------------------------------------------------
        # Buscamos la PRIMER vela que confirme la ruptura
        # --------------------------------------------------------------
        for _, row in df_win.iterrows():
            ts = row["timestamp"]
            close = float(row["close"])
            high = float(row.get("high", close))
            low = float(row.get("low", close))

            # Reglas de confirmación:
            if direction == "long":
                if close <= ref_price:
                    continue
                entry_price = close
                sl = entry_price * (1 - sl_buf)
            else:  # short
                if close >= ref_price:
                    continue
                entry_price = close
                sl = entry_price * (1 + sl_buf)

            # Cálculo mínimo de TP usando min_rr
            if direction == "long":
                one_r = entry_price - sl
                if one_r <= 0:
                    continue
                tp = entry_price + one_r * min_rr
            else:
                one_r = sl - entry_price
                if one_r <= 0:
                    continue
                tp = entry_price - one_r * min_rr

            if DEBUG_ENTRY:
                print(
                    f"[ENTRY DEBUG] FOUND at {ts}, entry={entry_price:.2f}, "
                    f"sl={sl:.2f}, tp={tp:.2f}, rr_min={min_rr}"
                )

            entry = Entry(
                direction=direction,
                timestamp=ts,
                entry_price=entry_price,
                sl=sl,
                tp=tp,
                trigger=trigger,
            )

            # ─────────────────────────────────────
            #  🔌 Supervisión IA (último filtro)
            # ─────────────────────────────────────
            # 🔹 Última regla: supervisor IA (si está activado)
            if ai_supervisor:
                decision = review_setup_with_ai(
                    symbol=trigger.symbol,
                    style=style,
                    impulse=trigger.impulse,
                    pullback=trigger.pullback,
                    trigger=trigger,
                    entry=entry,
                    dfs=dfs,
                )
                if not decision.approved:
                    # Podés loggear la razón si querés
                    print(
                        f"[AI_SUPERVISOR] Rechazó setup en {symbol} | "
                        f"dir={entry.direction} | motivo={decision.reason} "
                        f"(conf={decision.confidence:.2f})"
                    )
                    return None
            

            # Si no usamos IA o fue aprobado, devolvemos la entry normal
            return entry

        # No se encontró ninguna vela válida
        return None