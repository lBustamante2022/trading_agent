# app/ai/green_supervisor.py

import os
import json
from dataclasses import dataclass
from typing import Optional, List, Dict

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
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> List[dict]:
    """
    Ventana genérica de velas entre start_ts y end_ts,
    con campos compactos.
    """
    if df is None or df.empty:
        return []

    win = df[
        (df["timestamp"] >= start_ts) &
        (df["timestamp"] <= end_ts)
    ].copy().sort_values("timestamp")

    rows = []
    for _, r in win.iterrows():
        rows.append({
            "ts": r["timestamp"].isoformat(),
            "o": float(r["open"]),
            "h": float(r["high"]),
            "l": float(r["low"]),
            "c": float(r["close"]),
        })
    return rows


def review_setup_with_ai(
    symbol: str,
    style: GreenStyle,
    impulse: Impulse,
    pullback: Pullback,
    trigger: Trigger,
    entry: Entry,
    dfs: Dict[str, pd.DataFrame],
) -> AISupervisorDecision:
    """
    Envía un resumen del setup a OpenAI y devuelve aprobado / rechazado.

    Usa:
      - df de pullback_tf como “contexto HTF”
      - df de trigger_tf como “contexto LTF”
    """

    # --- elegir timeframes según el estilo ---
    df_pullback_tf = dfs.get(style.pullback_tf)
    df_trigger_tf = dfs.get(style.trigger_tf)

    # Obtenemos zona del pullback desde meta (si existe)
    zone_start = pullback.meta.get("zone_start", pullback.end_time)
    zone_end = pullback.meta.get("zone_end", pullback.end_time)

    if not isinstance(zone_start, pd.Timestamp):
        zone_start = pd.to_datetime(zone_start)
    if not isinstance(zone_end, pd.Timestamp):
        zone_end = pd.to_datetime(zone_end)

    # Ventana para "HTF" alrededor del pullback (ej. 40 velas antes, 10 después)
    # No sabemos la resolución exacta, así que usamos una aproximación en horas.
    ctx_pullback = _build_context_tf(
        df_pullback_tf,
        start_ts=zone_start - pd.Timedelta(hours=40),
        end_ts=zone_end + pd.Timedelta(hours=10),
    ) if df_pullback_tf is not None else []

    # Ventana para "LTF" alrededor del trigger
    ctx_trigger = _build_context_tf(
        df_trigger_tf,
        start_ts=trigger.timestamp - pd.Timedelta(hours=6),
        end_ts=trigger.timestamp + pd.Timedelta(hours=2),
    ) if df_trigger_tf is not None else []

    # Payload compacto con lo importante
    payload = {
        "symbol": symbol,
        "style": style.name,
        "direction": pullback.direction,
        "sr_level": impulse.sr_level,
        "impulse_start": impulse.start.isoformat(),
        "impulse_end": impulse.end.isoformat(),
        "pullback_zone_start": zone_start.isoformat(),
        "pullback_zone_end": zone_end.isoformat(),
        "trigger_time": trigger.timestamp.isoformat(),
        "trigger_ref_price": trigger.ref_price,
        "entry_time": entry.entry_time.isoformat(),
        "entry_sl": entry.sl,
        "entry_tp": entry.tp,
        "params": {
            "pullback_sr_tolerance": style.pullback_sr_tolerance,
            "pullback_min_swings": style.pullback_min_swings,
            "pullback_max_days": style.pullback_max_days,
            "min_rr": style.min_rr,
        },
        "context_pullback_tf": ctx_pullback,
        "context_trigger_tf": ctx_trigger,
    }

    system_prompt = """
Eres un trader profesional especializado en patrones de impulso + pullback + ruptura diagonal (estilo cuña) en criptomonedas.

Tu tarea es evaluar si un setup de trading es VÁLIDO o NO VÁLIDO según estas reglas:

1) IMPULSO:
   - Debe ser un movimiento direccional claro (no rango).
   - La dirección del pullback y del trigger debe ser coherente con ese impulso.

2) PULLBACK VÁLIDO:
   - Ocurre después del impulso y retrocede hacia el nivel de soporte/resistencia sr_level.
   - Debe tener ESTRUCTURA, no un simple "pincho":
     - Para LONG: máximos y mínimos descendentes que se acercan al sr_level.
     - Para SHORT: máximos y mínimos ascendentes que se acercan al sr_level.
   - Debe evitar:
     - Rebote en V sin estructura.
     - Que el precio haga un nuevo máximo/mínimo que invalide la lógica del impulso.

3) TRIGGER:
   - Debe romper una diagonal (cuña) del pullback cerca del final de éste.
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
        f"SETUP:\n```json\n{json.dumps(payload)}\n```"
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
        # fallback conservador: si algo sale mal, rechazamos
        return AISupervisorDecision(approved=False, confidence=0.0, reason="Error parseando respuesta IA")