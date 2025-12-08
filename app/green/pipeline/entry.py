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

    - SL:
          - Debe cubrir el mínimo/máximo de TODO el pullback
            (rango de precios del pullback) más un pequeño buffer (sl_buffer_pct).

    - TP:
          - SHORT → TP en el MÍNIMO precio desde que inició el pullback.
          - LONG  → TP en el MÁXIMO precio desde que inició el pullback.
          - El RR real (TP “natural” vs SL) debe ser >= min_rr.
            Si no se cumple, se descarta esa vela como entrada.

Devuelve 0 o 1 Entry por trigger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import builtins as _builtins

import pandas as pd
import time

from app.green.core import Entry, Trigger
from app.green.styles import GreenStyle
from app.ai.green_supervisor import review_setup_with_ai


# ----------------------------------------------------------------------
# DEBUG PRINT LOCAL
# ----------------------------------------------------------------------

DEBUG_ENTRY = False


def _log(*args, **kwargs):
    if DEBUG_ENTRY:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = " ".join(str(a) for a in args)
        _builtins.print(f"[{ts} DEBUG ENTRY] {msg}", **kwargs)


print = _log  # override local


def delay_ms(ms: int):
    """
    Pausa la ejecución la cantidad de milisegundos indicada.
    """
    time.sleep(ms / 1000.0)

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

    ai_supervisor:
      - Si es True, pasa el setup por el supervisor IA como última regla.
    """
    ai_supervisor: bool = False

    def detect_entry(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        trigger: Trigger,
    ) -> Optional[Entry]:

        tf = style.entry_tf
        df = dfs.get(tf)
        if df is None or df.empty:
            return None

        if "timestamp" not in df.columns or "close" not in df.columns:
            return None

        df = df.sort_values("timestamp").reset_index(drop=True)

        lookahead = int(getattr(cfg, "entry_max_minutes", 60))
        sl_buf = float(getattr(cfg, "entry_sl_buffer_pct", 0.002))
        min_rr = float(getattr(cfg, "entry_min_rr", 2.0))

        direction = trigger.direction
        start_ts = trigger.end
        end_ts = start_ts + timedelta(minutes=lookahead)

        pb_close = float(trigger.end_close)

        df_win = df[
            (df["timestamp"] >= start_ts) &
            (df["timestamp"] <= end_ts)
        ].copy()

        if df_win.empty:
            return None

        print(
            f"dir={direction}, tf={tf}, "
            f"start={start_ts}, end={end_ts}, pb_close={pb_close:.2f}"
        )

        # --------------------------------------------------------------
        # Info del pullback para SL y TP "natural"
        # --------------------------------------------------------------
        pb = trigger.pullback

        # Intentamos tomar el inicio del pullback como el primer swing.
        # swings = getattr(pb, "swings", [])
        # if swings:
        #     pb_start_ts = min(ts for (ts, _) in swings)
        # else:
        #     # Fallback: fin del impulso como inicio de pullback
        pb_start_ts = pb.impulse.end
        pb_end_ts = pb.end

        # Rango completo del pullback en el TF de entrada
        df_pb_range = df[
            (df["timestamp"] >= pb_start_ts) &
            (df["timestamp"] <= pb_end_ts)
        ].copy()

        pb_low = None
        pb_high = None
        if not df_pb_range.empty and "low" in df_pb_range.columns and "high" in df_pb_range.columns:
            pb_low = float(df_pb_range["low"].min())
            pb_high = float(df_pb_range["high"].max())

        print(
            f"PB RANGE [{pb_start_ts} → {pb_end_ts}] "
            f"pb_low={pb_low}, pb_high={pb_high}"
        )

        # --------------------------------------------------------------
        # Buscamos la PRIMER vela que confirme la ruptura
        # --------------------------------------------------------------
        for _, row in df_win.iterrows():
            ts = row["timestamp"]
            close = float(row["close"])
            high = float(row.get("high", close))
            low = float(row.get("low", close))

            # ---------------------------------------------
            # 1) Confirmación de entrada
            # ---------------------------------------------
            if direction == "long":
                if close <= pb_close:
                    continue
                entry_price = close
            else:  # short
                if close >= pb_close:
                    continue
                entry_price = close

            # ---------------------------------------------
            # 2) SL basado en TODO el pullback
            #    - LONG  → por debajo del mínimo low del pullback
            #    - SHORT → por encima del máximo high del pullback
            # ---------------------------------------------
            if pb_low is None or pb_high is None:
                # Sin info de rango → mejor no tomar la operación
                continue

            if direction == "long":
                base_sl = pb_low
                sl = base_sl * (1.0 - sl_buf)
            else:  # short
                base_sl = pb_high
                sl = base_sl * (1.0 + sl_buf)


            # Validar que el SL siga estando del lado correcto
            if direction == "long" and sl >= entry_price:
                continue
            if direction == "short" and sl <= entry_price:
                continue

            # ---------------------------------------------
            # 3) TP "natural" según inicio del pullback
            #
            #    - SHORT → TP en el mínimo precio desde inicio del pullback
            #    - LONG  → TP en el máximo precio desde inicio del pullback
            #      (usando el rango pb_low / pb_high)
            # ---------------------------------------------
            if pb_low is None or pb_high is None:
                # Sin info de rango → mejor no tomar la operación
                continue

            if direction == "long":
                tp_natural = pb_high
                # TP debe estar por encima de la entrada
                if tp_natural <= entry_price:
                    continue
                one_r = entry_price - sl
                if one_r <= 0:
                    continue
                rr = (tp_natural - entry_price) / one_r
                if rr < min_rr:
                    print(
                        f"LONG tp_natural RR insuficiente: "
                        f"entry={entry_price:.4f} tp={tp_natural:.4f} sl={sl:.4f} rr={rr:.2f} < min_rr={min_rr:.2f} → descarto"
                    )
                    continue
                tp = tp_natural
            else:  # short
                tp_natural = pb_low
                # TP debe estar por debajo de la entrada
                if tp_natural >= entry_price:
                    continue
                one_r = sl - entry_price
                if one_r <= 0:
                    continue
                rr = (entry_price - tp_natural) / one_r
                if rr < min_rr:
                    print(
                        f"SHORT tp_natural RR insuficiente: "
                        f"entry={entry_price:.4f} tp={tp_natural:.4f} sl={sl:.4f} rr={rr:.2f} < min_rr={min_rr:.2f} → descarto"
                    )
                    continue
                tp = tp_natural

            print(
                f"FOUND at {ts}, dir={direction}, "
                f"entry={entry_price:.2f}, sl={sl:.2f}, tp={tp:.2f}, "
                f"RR={rr:.2f} (min_rr={min_rr})"
            )

            entry = Entry(
                symbol=symbol,
                direction=direction,
                start=start_ts,
                end=ts,
                entry=entry_price,
                sl=sl,
                tp=tp,
                trigger=trigger,
            )

            # ─────────────────────────────────────
            # 4) Supervisor IA (último filtro opcional)
            # ─────────────────────────────────────
            if self.ai_supervisor:
                decision = review_setup_with_ai(
                    symbol=trigger.symbol,
                    style=style,
                    impulse=trigger.pullback.impulse,
                    pullback=trigger.pullback,
                    trigger=trigger,
                    entry=entry,
                    dfs=dfs,
                )
                if not decision.approved:
                    print(
                        f"ai supervisor: Rechazó setup en {symbol} | "
                        f"dir={entry.direction} | motivo={decision.reason} "
                        f"(conf={decision.confidence:.2f})"
                    )
                    delay_ms(1000)
                    return None

            # Si llegamos acá, la entrada es válida
            return entry

        # No se encontró ninguna vela válida
        return None