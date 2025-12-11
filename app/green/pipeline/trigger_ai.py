# app/green/pipeline/trigger_ai.py
"""
Trigger + Entry gobernados por IA para GREEN V3.

Idea:
    1. A partir de un Pullback válido construimos un ENTRY mecánico:
        - dirección = pullback.meta["direction"]
        - entry_time = pullback.meta["rebound_ts"] (o fin del pullback)
        - entry_price = close en esa vela
        - SL = un poco más allá del SR (buffer porcentual)
        - TP = entry + entry_min_rr * 1R

    2. Si use_ai == False → devolvemos ese Entry mecánico sin preguntar nada.

    3. Si use_ai == True → mandamos a OpenAI un resumen numérico del setup
       (impulso + pullback + entry propuesto) y pedimos un JSON:
            {
              "take_trade": true/false,
              "comment": "texto corto"
            }

       - Si take_trade == true → devolvemos el Entry.
       - Si false o hay error → devolvemos None (no operar ese pullback).

Dependencias:
    - Requiere tener OPENAI_API_KEY en el entorno si use_ai=True.
    - Soporta tanto el cliente nuevo `from openai import OpenAI` como el legacy
      `import openai` (elige automáticamente el que encuentre).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import timedelta
import os
import json
import builtins as _builtins

import numpy as np
import pandas as pd

from app.green.core import TriggerAIStrategy, Entry, Pullback
from app.green.styles import GreenStyle

# ----------------------------------------------------------------------
# INTENTO DE IMPORTAR OPENAI (nuevo o legacy)
# ----------------------------------------------------------------------


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")
_OPENAI_CLIENT_TYPE = None
_OPENAI_CLIENT = None

try:
    from openai import OpenAI  # type: ignore

    _OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY)
    _OPENAI_CLIENT_TYPE = "new"
except Exception:
    try:
        import openai  # type: ignore

        openai.api_key = OPENAI_API_KEY
        _OPENAI_CLIENT = openai
        _OPENAI_CLIENT_TYPE = "legacy"
    except Exception:
        _OPENAI_CLIENT = None
        _OPENAI_CLIENT_TYPE = None


# ----------------------------------------------------------------------
# DEBUG LOCAL
# ----------------------------------------------------------------------

DEBUG_TRIGGER_AI = True


def _log(*args, **kwargs):
    if DEBUG_TRIGGER_AI:
        from datetime import datetime

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = " ".join(str(a) for a in args)
        _builtins.print(f"[{ts} DEBUG TRIGGER_AI] {msg}", **kwargs)


print = _log  # override local


# ----------------------------------------------------------------------
# Estrategia Trigger + IA
# ----------------------------------------------------------------------


@dataclass
class DefaultTriggerAIStrategy(TriggerAIStrategy):
    """
    Estrategia que decide el ENTRY final y opcionalmente lo pasa por OpenAI.

    Atributos:
        use_ai: si True intenta validar cada setup con OpenAI.
        model:  nombre del modelo (nuevo stack) o legacy.
    """

    use_ai: bool = False

    # --------------------------------------------------------------
    # API PÚBLICA: decide_entry
    # --------------------------------------------------------------
    def decide_entry(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        pullback: Pullback,
    ) -> Optional[Entry]:
        """
        Devuelve un Entry mecánico, opcionalmente filtrado por IA.
        """

        # 1) Construir entry mecánico base
        entry = self._build_mechanical_entry(symbol, dfs, style, cfg, pullback)
        if entry is None:
            return None

        # 2) Sin IA → devolvemos entry tal cual
        if not self.use_ai:
            return entry

        # 3) Con IA → chequear entorno + cliente
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or _OPENAI_CLIENT is None or _OPENAI_CLIENT_TYPE is None:
            print("[AI] OPENAI_API_KEY no configurada o cliente OpenAI no disponible → fallback mecánico.")
            return entry

        # 4) Construir payload de features numéricas para el modelo
        features = self._build_features_for_ai(symbol, dfs, style, pullback, entry)

        # 5) Llamar a OpenAI
        decision = self._ask_openai_decision(features)
        if decision is None:
            # Ante error → usar entrada mecánica
            return entry

        take_trade = bool(decision.get("take_trade", True))
        comment = decision.get("comment", "")
        print(f"[AI] take_trade={take_trade} comment={comment}")

        if not take_trade:
            return None

        return entry

    # --------------------------------------------------------------
    # 1) Construcción ENTRY mecánico
    # --------------------------------------------------------------

    def _build_mechanical_entry(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        pullback: Pullback,
    ) -> Optional[Entry]:
        """
        Construye un Entry simple a partir del pullback:
            - entry_time = rebound_ts (o pb.end)
            - entry_price = close en esa vela (TF de pullback)
            - SL = más allá del SR con pequeño buffer
            - TP = entry + entry_min_rr * 1R
        """

        direction = pullback.meta.get("direction")
        if direction not in ("long", "short"):
            return None

        sr_level = float(pullback.meta.get("sr_level", np.nan))
        if not np.isfinite(sr_level) or sr_level <= 0:
            return None

        # Timeframe donde buscaremos el precio de la vela de entrada
        tf_pb = getattr(style, "pullback_tf", "15m")
        df_pb = dfs.get(tf_pb)
        if df_pb is None or df_pb.empty or "timestamp" not in df_pb.columns or "close" not in df_pb.columns:
            return None

        df_pb = df_pb.sort_values("timestamp").reset_index(drop=True)
        df_pb["timestamp"] = pd.to_datetime(df_pb["timestamp"])

        # Momento de entrada → usamos la vela de rebote si está, si no el fin del pullback
        entry_ts = pullback.meta.get("rebound_ts", pullback.meta.get("end"))
        if entry_ts is None:
            return None
        entry_ts = pd.to_datetime(entry_ts)

        # Buscamos la fila de esa vela (por timestamp exacto o la más cercana)
        row = df_pb[df_pb["timestamp"] == entry_ts]
        if row.empty:
            # buscamos la más cercana
            idx = (df_pb["timestamp"] - entry_ts).abs().idxmin()
            row = df_pb.iloc[[idx]]

        entry_price = float(row["close"].iloc[0])
        high = float(row.get("high", row["close"]).iloc[0])
        low = float(row.get("low", row["close"]).iloc[0])

        # Parámetros de estilo para SL/TP
        sl_buffer_pct = float(getattr(style, "entry_sl_buffer_pct", 0.001))
        min_rr = float(getattr(style, "entry_min_rr", 2.0))

        # Construcción de SL según dirección
        if direction == "long":
            # soporte: SL por debajo del SR
            sl = sr_level * (1.0 - sl_buffer_pct)
            # pero nunca por encima del mínimo de la vela
            sl = min(sl, low)
            one_r = entry_price - sl
            if one_r <= 0:
                return None
            tp = entry_price + min_rr * one_r
        else:  # short
            # resistencia: SL por encima del SR
            sl = sr_level * (1.0 + sl_buffer_pct)
            sl = max(sl, high)
            one_r = sl - entry_price
            if one_r <= 0:
                return None
            tp = entry_price - min_rr * one_r

        meta = {
            "symbol": symbol,
            "direction": direction,
            "start": entry_ts,
            "end": entry_ts,
            "entry": entry_price,
            "sl": sl,
            "tp": tp,
            # guardamos algunas cosas útiles
            "sr_level": sr_level,
            "pullback_start": pullback.meta.get("start"),
            "pullback_end": pullback.meta.get("end"),
            "rebound_ts": pullback.meta.get("rebound_ts"),
        }

        trigger_stub = {
            "direction": direction,
            "end": entry_ts,
        }

        trigger = type("TriggerStub", (), {"pullback": pullback, "meta": trigger_stub})

        entry_obj = Entry(
            trigger=trigger,  # type: ignore[arg-type]
            meta=meta,
        )
        return entry_obj

    # --------------------------------------------------------------
    # 2) Features para IA
    # --------------------------------------------------------------

    def _build_features_for_ai(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        pullback: Pullback,
        entry: Entry,
    ) -> Dict[str, Any]:
        """
        Prepara un diccionario compacto de datos del setup.
        Esto es lo que se envía a OpenAI como contexto.
        """

        imp = pullback.impulse
        pb = pullback.meta
        imp_meta = imp.meta
        e = entry.meta

        direction = pb.get("direction")
        sr_level = float(pb.get("sr_level", imp_meta.get("sr_level", np.nan)))
        sr_kind = pb.get("sr_kind", imp_meta.get("sr_kind", ""))

        features: Dict[str, Any] = {
            "symbol": symbol,
            "style_name": style.name,
            "direction": direction,
            "sr_kind": sr_kind,
            "sr_level": sr_level,
            "impulse": {
                "start": str(imp_meta.get("start")),
                "end": str(imp_meta.get("end")),
                "direction": imp_meta.get("direction"),
                "break_pct": imp_meta.get("break_pct", None),
                "range_pct": imp_meta.get("range_pct", None),
                "avg_body_impulse_pct": pb.get("avg_body_impulse_pct", None),
            },
            "pullback": {
                "start": str(pb.get("start")),
                "end": str(pb.get("end")),
                "sr_touch_ts": str(pb.get("sr_touch_ts")),
                "sr_touch_price": pb.get("sr_touch_price", None),
                "sr_touch_dist_pct": pb.get("sr_touch_dist_pct", None),
                "avg_body_pullback_pct": pb.get("avg_body_pullback_pct", None),
                "rebound_ts": str(pb.get("rebound_ts")),
                "rebound_dir": pb.get("rebound_dir", None),
                "rebound_body_ratio": pb.get("rebound_body_ratio", None),
            },
            "entry": {
                "time": str(e.get("end")),
                "price": float(e.get("entry")),
                "sl": float(e.get("sl")),
                "tp": float(e.get("tp")),
                "risk_R": 1.0,
                "planned_rr": float(
                    (e.get("tp") - e.get("entry"))
                    / (e.get("entry") - e.get("sl"))
                )
                if direction == "long"
                else float(
                    (e.get("entry") - e.get("tp"))
                    / (e.get("sl") - e.get("entry"))
                ),
            },
            "style_params": {
                "pullback_sr_tolerance": getattr(style, "pullback_sr_tolerance", None),
                "pullback_slow_body_factor": getattr(style, "pullback_slow_body_factor", None),
                "pullback_rebound_body_min_ratio": getattr(style, "pullback_rebound_body_min_ratio", None),
                "entry_sl_buffer_pct": getattr(style, "entry_sl_buffer_pct", None),
                "entry_min_rr": getattr(style, "entry_min_rr", None),
            },
        }

        # Pequeño resumen de precio alrededor del pullback (no velas crudas, solo stats)
        tf_pb = getattr(style, "pullback_tf", "15m")
        df_pb = dfs.get(tf_pb)
        if df_pb is not None and not df_pb.empty and "timestamp" in df_pb.columns and "close" in df_pb.columns:
            df_pb = df_pb.sort_values("timestamp").reset_index(drop=True)
            df_pb["timestamp"] = pd.to_datetime(df_pb["timestamp"])

            pb_start = pd.to_datetime(pb.get("start"))
            pb_end = pd.to_datetime(pb.get("end"))

            win_start = pb_start - timedelta(hours=24)
            win_end = pb_end + timedelta(hours=24)
            seg = df_pb[
                (df_pb["timestamp"] >= win_start)
                & (df_pb["timestamp"] <= win_end)
            ].copy()
            if not seg.empty:
                closes = seg["close"].astype(float)
                features["context_price_stats"] = {
                    "tf": tf_pb,
                    "rows": int(len(seg)),
                    "min_close": float(closes.min()),
                    "max_close": float(closes.max()),
                    "mean_close": float(closes.mean()),
                    "std_close": float(closes.std(ddof=0)),
                }

        return features

    # --------------------------------------------------------------
    # 3) Llamada a OpenAI
    # --------------------------------------------------------------

    def _ask_openai_decision(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Envía los features a OpenAI y espera un JSON:
            { "take_trade": bool, "comment": "..." }
        """
        if _OPENAI_CLIENT is None or _OPENAI_CLIENT_TYPE is None:
            return None

        system_msg = (
            "Eres un asistente de trading cuantitativo. "
            "Recibirás la descripción NUMÉRICA de un posible trade de pullback contra soporte/resistencia. "
            "Debes decidir si el trade es coherente con la idea de: impulso fuerte, pullback lento hasta SR y "
            "rebote claro a favor del impulso. Responde SOLO con un JSON válido "
            'de la forma {"take_trade": bool, "comment": "texto corto"}.'
        )

        user_msg = (
            "Analiza el siguiente setup y decide si tomarías el trade:\n\n"
            + json.dumps(features, ensure_ascii=False, indent=2)
        )

        try:
            if _OPENAI_CLIENT_TYPE == "new":
                # Nuevo cliente (openai>=1, from openai import OpenAI)
                resp = _OPENAI_CLIENT.chat.completions.create(
                    model= OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "trade_decision",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "take_trade": {"type": "boolean"},
                                    "comment": {"type": "string"},
                                },
                                "required": ["take_trade"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    temperature=0.0,
                )
                content = resp.choices[0].message.content
            else:
                # Cliente legacy (openai.ChatCompletion)
                resp = _OPENAI_CLIENT.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                )
                content = resp["choices"][0]["message"]["content"]

            if not content:
                return None

            decision = json.loads(content)
            if not isinstance(decision, dict):
                return None
            return decision
        except Exception as e:
            print(f"[AI] Error llamando a OpenAI: {e}")
            return None