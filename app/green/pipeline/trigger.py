# app/green/pipeline/trigger.py
"""
Detección de TRIGGER para GREEN v3 (simplificado).

Responsabilidad:

    A partir de un Pullback ya válido (cuña contra SR),
    buscar la primera vela "fuerte" que rompa la EMA en la
    dirección del trade.

Lógica:

    - Usamos el TF de trigger: style.trigger_tf
    - Ventana temporal:
          desde pullback.meta["end"] (inclusive)
          hasta pullback.meta["end"] + trigger_max_break_minutes

    - SR:
          sr_level = impulse.meta["sr_level"]

    - INVALIDACIÓN (rompe fuerte el SR en contra):
        LONG  (desde soporte):
            Si (sr_level - close) / sr_level > break_tol  → setup INVALIDADO.
        SHORT (desde resistencia):
            Si (close - sr_level) / sr_level > break_tol  → setup INVALIDADO.

    - TRIGGER (vela de entrada):

        Calculamos una EMA corta sobre los cierres de la ventana:

            ema_len = trigger_entry_ema_length (default 3)

        LONG:
            - Buscamos la PRIMER vela que cumpla:
                * ema ya calculada (no NaN)
                * close >= ema * (1 + ema_break_buffer_pct)
                * close > open  (vela alcista)
                * body_ratio = |close - open| / (high - low) >= body_min_ratio
                  (ej. 0.6 → 60% cuerpo).

        SHORT:
            - Buscamos la PRIMER vela que cumpla:
                * ema ya calculada (no NaN)
                * close <= ema * (1 - ema_break_buffer_pct)
                * close < open  (vela bajista)
                * body_ratio >= body_min_ratio.

    Si se encuentra esa vela, se construye un Trigger que ya representa
    esa ruptura calificada. Entry luego usa:

        trigger.meta["end"]       → timestamp de la vela
        trigger.meta["end_close"] → close de la vela (precio "natural" de entrada).

Parámetros relevantes (normalmente en el style):

    - trigger_tf: str
    - trigger_max_break_minutes: int (ej. 36*60)
    - trigger_entry_ema_length: int (default 3)
    - trigger_entry_body_min_ratio: float (default 0.6)
    - trigger_entry_ema_break_buffer_pct: float (default 0.0)

    - pullback_min_tolerance / pullback_sr_tolerance para invalidación SR.
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


# Sobrescribimos print localmente para debug controlado
print = _log  # type: ignore


# ----------------------------------------------------------------------
# Estrategia concreta de TRIGGER
# ----------------------------------------------------------------------

@dataclass
class DefaultTriggerStrategy(TriggerStrategy):
    """
    Estrategia de TRIGGER simplificada:

        - Invalida el setup si el precio rompe el SR en contra.
        - Busca la primera vela que:
            * rompa la EMA en la dirección del trade, y
            * tenga cuerpo >= X% del rango (body_min_ratio).
    """

    def detect_triggers(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        pullback: Pullback,
    ) -> List[Trigger]:

        # ------------------------------------------------------------------
        # 1) Validaciones básicas de DataFrame
        # ------------------------------------------------------------------
        tf = style.trigger_tf
        df = dfs.get(tf)
        if df is None or df.empty:
            return []

        required_cols = {"timestamp", "open", "high", "low", "close"}
        if not required_cols.issubset(df.columns):
            print(
                f"DF de {tf} no tiene las columnas necesarias "
                f"({required_cols}). No se detectan triggers."
            )
            return []

        df = df.sort_values("timestamp").reset_index(drop=True)

        # ------------------------------------------------------------------
        # 2) Datos base desde Pullback / Impulse / Config
        # ------------------------------------------------------------------
        direction = pullback.meta["direction"]  # "long" / "short"
        sr_level = float(pullback.impulse.meta["sr_level"])

        pb_end_ts = pullback.meta["end"]
        if not isinstance(pb_end_ts, pd.Timestamp):
            pb_end_ts = pd.to_datetime(pb_end_ts)

        # Ventana máxima de vida del trigger
        max_minutes = int(getattr(style, "trigger_max_break_minutes", 36 * 60))

        # Tolerancia de INVALIDACIÓN del setup si se rompe el SR en contra.
        break_tol = float(
            getattr(
                style,
                "trigger_sr_tolerance",
                getattr(style, "pullback_sr_tolerance", 0.05),
            )
        )

        # Parámetros de la EMA y del cuerpo de la vela
        ema_len = int(getattr(style, "trigger_entry_ema_length", 3))
        body_min_ratio = float(getattr(style, "trigger_entry_body_min_ratio", 0.6))
        ema_break_buffer_pct = float(
            getattr(style, "trigger_entry_ema_break_buffer_pct", 0.0)
        )

        # Ventana temporal donde buscamos la vela de ruptura
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

        # Calculamos EMA corta de los cierres en la ventana
        df_win["ema"] = df_win["close"].ewm(span=ema_len, adjust=False).mean()

        print(
            f"TRIGGER: dir={direction}, tf={tf}, "
            f"pb_end={pb_end_ts}, sr_level={sr_level:.2f}, "
            f"ema_len={ema_len}, body_min={body_min_ratio:.2f}, "
            f"break_tol={break_tol:.4f}, max_minutes={max_minutes}"
        )

        triggers: List[Trigger] = []

        # ------------------------------------------------------------------
        # 3) Lógica LONG / SHORT con invalidación + EMA + 60% cuerpo
        # ------------------------------------------------------------------
        for _, row in df_win.iterrows():
            ts = row["timestamp"]
            if not isinstance(ts, pd.Timestamp):
                ts = pd.to_datetime(ts)

            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])
            c = float(row["close"])
            ema = float(row["ema"])

            # Si todavía no tenemos EMA estable (primeras velas), seguimos.
            if pd.isna(ema):
                continue

            # Rango y cuerpo de la vela
            rang = max(h - l, 1e-9)  # para evitar división por cero
            body = abs(c - o)
            body_ratio = body / rang

            # --------------------------------------------------------------
            # 3.1 INVALIDACIÓN por ruptura del SR
            # --------------------------------------------------------------
            if direction == "long":
                # ¿rompió el soporte demasiado hacia abajo?
                dist_break = (sr_level - c) / sr_level
                if dist_break > break_tol:
                    print(
                        f"LONG invalidado: ts={ts}, close={c:.2f} "
                        f"rompe soporte más allá de tolerancia "
                        f"(dist_break={dist_break:.4f} > break_tol={break_tol:.4f})"
                    )
                    return []

            elif direction == "short":
                # ¿rompió la resistencia demasiado hacia arriba?
                dist_break = (c - sr_level) / sr_level
                if dist_break > break_tol:
                    print(
                        f"SHORT invalidado: ts={ts}, close={c:.2f} "
                        f"rompe resistencia más allá de tolerancia "
                        f"(dist_break={dist_break:.4f} > break_tol={break_tol:.4f})"
                    )
                    return []

            # --------------------------------------------------------------
            # 3.2 BÚSQUEDA DE VELA "BUENA" (EMA + cuerpo + dirección)
            # --------------------------------------------------------------
            if direction == "long":
                # Vela alcista fuerte que rompe EMA hacia arriba
                cond_break_ema = c >= ema * (1.0 + ema_break_buffer_pct)
                cond_dir = c > o
                cond_body = body_ratio >= body_min_ratio

                if cond_break_ema and cond_dir and cond_body:
                    print(
                        f"LONG trigger: ts={ts}, close={c:.2f}, "
                        f"ema={ema:.2f}, body_ratio={body_ratio:.3f}"
                    )
                    tr = Trigger(
                        pullback=pullback,
                        meta={
                            "symbol": symbol,
                            "direction": "long",
                            "start": start_ts,
                            "end": ts,
                            "end_close": c,
                        },
                    )
                    triggers.append(tr)
                    break  # solo el primero

            elif direction == "short":
                # Vela bajista fuerte que rompe EMA hacia abajo
                cond_break_ema = c <= ema * (1.0 - ema_break_buffer_pct)
                cond_dir = c < o
                cond_body = body_ratio >= body_min_ratio

                if cond_break_ema and cond_dir and cond_body:
                    print(
                        f"SHORT trigger: ts={ts}, close={c:.2f}, "
                        f"ema={ema:.2f}, body_ratio={body_ratio:.3f}"
                    )
                    tr = Trigger(
                        pullback=pullback,
                        meta={
                            "symbol": symbol,
                            "direction": "short",
                            "start": start_ts,
                            "end": ts,
                            "end_close": c,
                        },
                    )
                    triggers.append(tr)
                    break  # solo el primero

        # En esta versión devolvemos 0 o 1 trigger por pullback.
        return triggers







# """
# Trigger simplificado para GREEN v3:

#     - A partir de un Pullback válido, buscamos una ÚNICA vela de entrada.
#     - Esa vela debe ser:
#         * Envolvente: su rango (high/low) se "come" las últimas N velas previas.
#         * Con cuerpo ≥ X% del rango total (por defecto 60%).
#         * Con dirección coherente con el trade:
#               - LONG  → vela alcista (close > open)
#               - SHORT → vela bajista (close < open)

#     - Además, mantenemos la INVALIDACIÓN por ruptura del SR en contra:
#           LONG  → si (sr_level - close) / sr_level > break_tol → setup inválido
#           SHORT → si (close - sr_level) / sr_level > break_tol → setup inválido

# Parámetros (tomados de style / cfg):

#     - style.trigger_tf                  → timeframe para buscar la vela.
#     - style.trigger_max_break_minutes   → ventana máxima desde pullback.end.
#     - style.trigger_entry_body_min_ratio (opcional, default 0.6)
#           → porcentaje mínimo de cuerpo (ej. 0.6 = 60%).
#     - style.trigger_engulf_bars (opcional, default 2)
#           → cantidad de velas que debe engullir (2 o 3 típicamente).

#     - style.pullback_min_tolerance / pullback_sr_tolerance
#           → tolerancia para invalidar el SR en contra.

# Meta de Pullback (esperado):

#     pullback.meta:
#         - "direction": "long" / "short"
#         - "start": datetime
#         - "end": datetime
#         - "end_close": float

#     pullback.impulse.meta:
#         - "sr_level": float (soporte/resistencia)

# Meta de Trigger generado:

#     trigger.meta:
#         - "symbol": str
#         - "direction": "long" / "short"
#         - "start": datetime  (igual a pullback.meta["end"])
#         - "end": datetime    (timestamp de la vela envolvente)
#         - "end_close": float (close de esa vela = precio de entrada natural)
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List, Any

# from datetime import datetime, timedelta
# import builtins as _builtins

# import pandas as pd

# from app.green.core import TriggerStrategy, Trigger, Pullback
# from app.green.styles import GreenStyle

# # ----------------------------------------------------------------------
# # DEBUG
# # ----------------------------------------------------------------------

# DEBUG_TRIGGER = False


# def _log(*args, **kwargs):
#     if DEBUG_TRIGGER:
#         ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         msg = " ".join(str(a) for a in args)
#         _builtins.print(f"[{ts} DEBUG TRIGGER] {msg}", **kwargs)


# print = _log  # override local


# # ----------------------------------------------------------------------
# # Estrategia TRIGGER simplificada
# # ----------------------------------------------------------------------

# @dataclass
# class DefaultTriggerStrategy(TriggerStrategy):

#     def detect_triggers(
#         self,
#         symbol: str,
#         dfs: Dict[str, pd.DataFrame],
#         style: GreenStyle,
#         cfg: Any,
#         pullback: Pullback,
#     ) -> List[Trigger]:

#         tf = style.trigger_tf
#         df = dfs.get(tf)
#         if df is None or df.empty:
#             return []

#         required_cols = {"timestamp", "open", "high", "low", "close"}
#         if not required_cols.issubset(df.columns):
#             print(f"Faltan columnas en TF {tf}: {required_cols}")
#             return []

#         df = df.sort_values("timestamp").reset_index(drop=True)

#         # ------------------------------------------------------------------
#         # Datos base desde Pullback / Impulse / Style
#         # ------------------------------------------------------------------
#         direction = pullback.meta["direction"]  # "long" / "short"
#         sr_level = float(pullback.impulse.meta["sr_level"])
#         pb_end_ts = pullback.meta["end"]

#         max_minutes = int(getattr(style, "trigger_max_break_minutes", 36 * 60))

#         # porcentaje mínimo de cuerpo (ej: 0.6 = 60%)
#         body_min_ratio = float(getattr(style, "trigger_entry_body_min_ratio", 0.6))

#         # cantidad de velas a engullir (2 o 3 típicamente)
#         engulf_bars = int(getattr(style, "trigger_engulf_bars", 2))
#         if engulf_bars < 1:
#             engulf_bars = 1

#         # tolerancia de invalidación del SR
#         break_tol = float(
#             getattr(
#                 style,
#                 "trigger_sr_tolerance",
#                 getattr(style, "trigger_sr_tolerance", 0.05),
#             )
#         )

#         # Ventana temporal donde buscamos la vela envolvente
#         start_ts = pb_end_ts
#         end_ts = pb_end_ts + timedelta(minutes=max_minutes)

#         df_win = df[
#             (df["timestamp"] >= start_ts) &
#             (df["timestamp"] <= end_ts)
#         ].copy()

#         if df_win.empty:
#             print(
#                 f"Ventana vacía ({start_ts} → {end_ts}) "
#                 f"para TF {tf}"
#             )
#             return []

#         print(
#             f"dir={direction}, tf={tf}, "
#             f"pb_end={pb_end_ts}, sr_level={sr_level:.2f}, "
#             f"body_min={body_min_ratio:.2f}, engulf_bars={engulf_bars}, "
#             f"break_tol={break_tol:.4f}, max_minutes={max_minutes}, df={len(df_win)}"
#         )

#         triggers: List[Trigger] = []

#         # ------------------------------------------------------------------
#         # Recorremos vela a vela, buscando la primera envolvente válida
#         # ------------------------------------------------------------------
#         df_win = df_win.reset_index(drop=True)

#         for idx in range(len(df_win)):
#             row = df_win.iloc[idx]
#             ts = row["timestamp"]
#             o = float(row["open"])
#             h = float(row["high"])
#             l = float(row["low"])
#             c = float(row["close"])

#             # Rango / cuerpo
#             rang = max(h - l, 1e-9)
#             body = abs(c - o)
#             body_ratio = body / rang

#             # --------------------------------------------------------------
#             # 1) INVALIDACIÓN por ruptura del SR en contra
#             # --------------------------------------------------------------
#             if direction == "long":
#                 # Si el close se aleja DEMASIADO por debajo del soporte
#                 dist_break = (sr_level - c) / sr_level
#                 if dist_break > break_tol:
#                     print(
#                         f"LONG invalidado: ts={ts}, close={c:.2f}, "
#                         f"dist_break={dist_break:.4f} > break_tol={break_tol:.4f}"
#                     )
#                     return []
#             else:  # short
#                 # Si el close se aleja DEMASIADO por encima de la resistencia
#                 dist_break = (c - sr_level) / sr_level
#                 if dist_break > break_tol:
#                     print(
#                         f"SHORT invalidado: ts={ts}, close={c:.2f}, "
#                         f"dist_break={dist_break:.4f} > break_tol={break_tol:.4f}"
#                     )
#                     return []

#             # Hasta acá, el setup sigue válido. Ahora vemos si ESTA vela
#             # es una buena vela de entrada.

#             # Necesitamos al menos "engulf_bars" velas previas
#             if idx < engulf_bars:
#                 continue

#             prev = df_win.iloc[idx - engulf_bars:idx]
#             prev_high = float(prev["high"].max())
#             prev_low = float(prev["low"].min())

#             # Condiciones de envolvente:
#             #   - su rango (high/low) se come el rango de las últimas N velas
#             cond_engulf = (h >= prev_high) and (l <= prev_low)

#             # Dirección y cuerpo
#             if direction == "long":
#                 cond_dir = c > o          # vela alcista
#             else:
#                 cond_dir = c < o          # vela bajista

#             cond_body = body_ratio >= body_min_ratio

#             # print(f"cond_engulf={cond_engulf} cond_dir={cond_dir} cond_body={cond_body}")
#             if cond_engulf and cond_dir and cond_body:
#                 print(
#                     f"{direction.upper()} envolvente OK @ {ts} | "
#                     f"c={c:.2f}, body_ratio={body_ratio:.2f}, "
#                     f"prev_high={prev_high:.2f}, prev_low={prev_low:.2f}"
#                 )

#                 tr = Trigger(
#                     pullback=pullback,
#                     meta={
#                         "symbol": symbol,
#                         "direction": direction,
#                         "start": start_ts,
#                         "end": ts,
#                         "end_close": c,
#                     },
#                 )
#                 triggers.append(tr)
#                 break  # solo tomamos la primera vela válida
                
#         return triggers