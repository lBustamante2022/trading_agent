# app/ai/green_supervisor.py

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import pandas as pd
from openai import OpenAI

from app.green.core import Impulse, Pullback, Trigger, Entry
from app.green.styles import GreenStyle

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class AISupervisorDecision:
    approved: bool
    reason: str
    confidence: float


def _build_context_tf(
    df: pd.DataFrame,
    center_ts: pd.Timestamp,
    lookback_bars: int = 80,
    forward_bars: int = 20,
) -> List[dict]:
    """
    Crea una ventana de velas alrededor de un timestamp de referencia.
    No asume timeframe fijo, solo usa cantidad de velas.
    """
    if df is None or df.empty or "timestamp" not in df.columns:
        return []

    df = df.sort_values("timestamp").reset_index(drop=True)

    before = df[df["timestamp"] <= center_ts].tail(lookback_bars)
    after = df[df["timestamp"] > center_ts].head(forward_bars)

    win = pd.concat([before, after], ignore_index=True)
    rows: List[dict] = []

    for _, r in win.iterrows():
        rows.append(
            {
                "ts": pd.to_datetime(r["timestamp"]).isoformat(),
                "o": float(r["open"]),
                "h": float(r["high"]),
                "l": float(r["low"]),
                "c": float(r["close"]),
            }
        )

    return rows


def review_setup_with_ai(
    symbol: str,
    cfg: Any,                     # GreenConfig o similar
    style: GreenStyle,
    impulse: Impulse,
    pullback: Pullback,
    trigger: Trigger,
    entry: Entry,
    dfs: Dict[str, pd.DataFrame],
) -> AISupervisorDecision:
    """
    Versión GREEN v3 del supervisor IA.

    Usa:
      - TF de pullback (style.pullback_tf) para contexto 'pullback_tf'
      - TF de trigger (style.trigger_tf) para contexto 'trigger_tf'
    """

    # Si no hay API key, aprobamos todo (modo "apagado")
    if not os.getenv("OPENAI_API_KEY"):
        return AISupervisorDecision(
            approved=True,
            confidence=0.0,
            reason="OPENAI_API_KEY no definido, supervisor IA desactivado.",
        )

    df_pb = dfs.get(style.pullback_tf)
    df_tg = dfs.get(style.trigger_tf)

    ctx_pullback = _build_context_tf(
        df=df_pb,
        center_ts=pullback.end_time,
        lookback_bars=80,
        forward_bars=20,
    )
    ctx_trigger = _build_context_tf(
        df=df_tg,
        center_ts=trigger.timestamp,
        lookback_bars=80,
        forward_bars=20,
    )

    payload = {
        "symbol": symbol,
        "direction": pullback.direction,
        "sr_level": impulse.sr_level,
        "impulse_start": impulse.start.isoformat(),
        "impulse_end": impulse.end.isoformat(),
        "pullback_end_time": pullback.end_time.isoformat(),
        "trigger_time": trigger.timestamp.isoformat(),
        "entry_time": entry.entry_time.isoformat(),
        "entry_sl": entry.sl,
        "entry_tp": entry.tp,
        "cfg": {
            # Solo algunos parámetros clave; si alguno no existe, usamos getattr con default
            "channel_width_pct": float(getattr(cfg, "channel_width_pct", 0.0)),
            "pullback_sr_tolerance": float(getattr(cfg, "pullback_sr_tolerance", 0.0)),
            "pullback_min_swings": int(getattr(cfg, "pullback_min_swings", 0)),
            "pullback_max_days": float(getattr(cfg, "pullback_max_days", 0.0)),
            "min_rr": float(getattr(cfg, "min_rr", 0.0)),
        },
        "style": {
            "name": style.name,
            "impulse_tf": style.impulse_tf,
            "pullback_tf": style.pullback_tf,
            "trigger_tf": style.trigger_tf,
            "entry_tf": style.entry_tf,
        },
        "context_pullback_tf": ctx_pullback,
        "context_trigger_tf": ctx_trigger,
    }

    system_prompt = """
Eres un trader profesional especializado en patrones de:
  - impulso direccional
  - pullback estructurado hacia un nivel de soporte/resistencia
  - ruptura (trigger) en forma de cuña / canal diagonal.

Tu tarea es evaluar si un setup de trading es VÁLIDO o NO VÁLIDO según estas reglas:

1) IMPULSO:
   - Debe ser un movimiento direccional claro (no rango plano).
   - La dirección del pullback y del trigger debe ser coherente con ese impulso.

2) PULLBACK VÁLIDO:
   - Ocurre después del impulso, volviendo hacia un nivel lógico (sr_level).
   - Debe tener ESTRUCTURA, no un simple "pincho":
     - Para LONG: máximos y mínimos descendentes que se acercan al sr_level.
     - Para SHORT: máximos y mínimos ascendentes que se acercan al sr_level.
   - Debe evitar:
     - Rebote en V sin estructura.
     - Que el precio haga un nuevo máximo/mínimo que invalide la lógica del impulso.

3) TRIGGER:
   - Debe romper una diagonal del pullback cerca del final de éste.
   - Debe estar razonablemente cerca del sr_level y del "final" del pullback.
   - Debe ser coherente con la dirección del trade (no en contra).

4) ENTRY:
   - Debe estar próximo en el tiempo al trigger (pocas velas después).
   - SL y TP deben tener sentido con la estructura del impulso y el pullback.

Debes devolver SIEMPRE un JSON EXACTO con este formato:

{
  "approved": true/false,
  "confidence": float entre 0 y 1,
  "reason": "texto corto explicando por qué"
}

No devuelvas nada más fuera de ese JSON.
"""

    user_prompt = (
        "Analiza este setup de trading de forma estricta. "
        "Solo apruébalo si el pullback realmente se parece a una cuña clara hacia sr_level "
        "y el trigger está en un punto lógico de ruptura.\n\n"
        f"SETUP:\n```json\n{json.dumps(payload, default=str)}\n```"
    )

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
    )

    msg = resp.choices[0].message.content
    try:
        data = json.loads(msg)
        approved = bool(data.get("approved", False))
        confidence = float(data.get("confidence", 0.0))
        reason = str(data.get("reason", ""))
        return AISupervisorDecision(approved=approved, confidence=confidence, reason=reason)
    except Exception:
        # fallback conservador
        return AISupervisorDecision(
            approved=False,
            confidence=0.0,
            reason="Error parseando respuesta IA",
        )