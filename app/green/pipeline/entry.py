# ================================================================
# ENTRY para GREEN V3 — versión mejorada
#
# Cambios clave:
#   ✔ SL robusto basado en TODO el pullback
#   ✔ RR mínimo pero sin TP "artificial"
#   ✔ COOLDOWN fuerte por pullback (set interno)
#   ✔ mantiene API original para el pipeline
# ================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime
import time
import builtins as _builtins

import pandas as pd

from app.green.core import Trigger, Entry
from app.green.styles import GreenStyle
from app.ai.green_supervisor import review_setup_with_ai

DEBUG_ENTRY = False


def _log(*args, **kwargs):
    if DEBUG_ENTRY:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = " ".join(str(a) for a in args)
        _builtins.print(f"[{ts} DEBUG ENTRY] {msg}", **kwargs)


print = _log  # override


def delay_ms(ms: int):
    time.sleep(ms / 1000.0)


# ================================================================
# ENTRY STRATEGY
# ================================================================

@dataclass
class DefaultEntryStrategy:
    """
    Estrategia genérica de ENTRY posterior al Trigger.

    - No espera más velas: Trigger ya eligió la vela "buena".
    - Calcula SL robusto usando TODO el pullback.
    - Valida un RR mínimo usando un TP "natural" (estructura).
    - Implementa cooldown FUERTE por pullback:
        un mismo rango de pullback solo puede generar UNA entrada.
    """

    ai_supervisor: bool = False

    # set de IDs de pullback ya usados (para todo el backtest/run)
    used_pullbacks: set[str] = field(default_factory=set, repr=False)

    def detect_entry(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        trigger: Trigger,
    ) -> Optional[Entry]:

        pb = trigger.pullback
        impulse = pb.impulse
        direction = trigger.meta["direction"]

        # ------------------------------------------------------------
        # Identificador único del pullback (cooldown fuerte)
        # ------------------------------------------------------------
        pb_start_ts = impulse.meta.get("end", pb.meta["start"])
        pb_end_ts = pb.meta["end"]

        if not isinstance(pb_start_ts, pd.Timestamp):
            pb_start_ts = pd.to_datetime(pb_start_ts)
        if not isinstance(pb_end_ts, pd.Timestamp):
            pb_end_ts = pd.to_datetime(pb_end_ts)

        pb_id = f"{symbol}|{direction}|{pb_start_ts.isoformat()}|{pb_end_ts.isoformat()}"

        if pb_id in self.used_pullbacks:
            print(f"ENTRY descartado: pullback ya operado ({pb_id}).")
            return None

        # Marcamos como usado
        self.used_pullbacks.add(pb_id)

        # (Seguimos manteniendo el flag en meta por si ayuda a debug)
        pb.meta["_entry_used"] = True

        # ------------------------------------------------------------
        # Datos de mercado
        # ------------------------------------------------------------
        tf = style.entry_tf
        df = dfs.get(tf)
        if df is None or df.empty:
            return None

        df = df.sort_values("timestamp").reset_index(drop=True)

        entry_time = trigger.meta["end"]
        entry_price = float(trigger.meta["end_close"])

        if not isinstance(entry_time, pd.Timestamp):
            entry_time = pd.to_datetime(entry_time)

        # Parámetros
        sl_buf = float(getattr(cfg, "entry_sl_buffer_pct", 0.002))
        min_rr = float(getattr(cfg, "entry_min_rr", 2.0))

        # ------------------------------------------------------------
        # RANGO del PULLBACK para construir SL robusto
        # ------------------------------------------------------------
        df_pb = df[
            (df["timestamp"] >= pb_start_ts) &
            (df["timestamp"] <= pb_end_ts)
        ].copy()

        if df_pb.empty or "low" not in df_pb.columns or "high" not in df_pb.columns:
            print("ENTRY descartado: no se pudo obtener rango del pullback.")
            return None

        pb_low = float(df_pb["low"].min())
        pb_high = float(df_pb["high"].max())

        print(f"PB RANGE: low={pb_low:.2f}, high={pb_high:.2f}")

        # ------------------------------------------------------------
        # SL robusto
        # ------------------------------------------------------------
        if direction == "long":
            base_sl = pb_low
            sl = base_sl * (1.0 - sl_buf)
            if sl >= entry_price:
                print("SL inválido en LONG (sl >= entry).")
                return None
        else:  # short
            base_sl = pb_high
            sl = base_sl * (1.0 + sl_buf)
            if sl <= entry_price:
                print("SL inválido en SHORT (sl <= entry).")
                return None

        # Distancia 1R
        if direction == "long":
            one_r = entry_price - sl
        else:
            one_r = sl - entry_price

        if one_r <= 0:
            print("ENTRY descartado: 1R no positivo.")
            return None

        # ------------------------------------------------------------
        # TP NATURAL para validar RR mínimo
        # (Trailing real se maneja luego en Position)
        # ------------------------------------------------------------
        if direction == "long":
            tp_nat = pb_high
            if tp_nat <= entry_price:
                print("TP natural inválido para LONG.")
                return None
            rr = (tp_nat - entry_price) / one_r
        else:
            tp_nat = pb_low
            if tp_nat >= entry_price:
                print("TP natural inválido para SHORT.")
                return None
            rr = (entry_price - tp_nat) / one_r

        # ------------------------------------------------------------
        # Validar RR mínimo
        # ------------------------------------------------------------
        if rr < min_rr:
            print(
                f"ENTRY descartado por RR insuficiente: rr={rr:.2f} < min_rr={min_rr}"
            )
            return None

        tp = tp_nat  # TP de referencia; trailing se encargará después

        print(
            f"ENTRY OK: dir={direction} entry={entry_price:.2f} sl={sl:.2f} "
            f"tp_nat={tp:.2f} RR={rr:.2f}"
        )

        entry = Entry(
            trigger=trigger,
            meta={
                "symbol": symbol,
                "direction": direction,
                "start": trigger.meta.get("start", pb.meta.get("start", pb_start_ts)),
                "end": entry_time,
                "entry": entry_price,
                "sl": sl,
                "tp": tp,
            },
        )

        # ------------------------------------------------------------
        # Supervisor IA (opcional)
        # ------------------------------------------------------------
        if self.ai_supervisor:
            decision = review_setup_with_ai(
                symbol=symbol,
                style=style,
                impulse=pb.impulse,
                pullback=pb,
                trigger=trigger,
                entry=entry,
                dfs=dfs,
            )
            if not decision.approved:
                print(
                    f"AI Rechaza ENTRY: motivo={decision.reason}, "
                    f"conf={decision.confidence:.2f}"
                )
                return None

        return entry