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
model = os.getenv("OPENAI_MODEL")

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
    # zone_start = pullback.meta.get("zone_start", pullback.end_time)
    # zone_end = pullback.meta.get("zone_end", pullback.end_time)
    zone_start = impulse.end
    zone_end = pullback.end

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
        start_ts=trigger.end - pd.Timedelta(hours=6),
        end_ts=trigger.end + pd.Timedelta(hours=2),
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
        "trigger_time": trigger.end.isoformat(),
        "trigger_ref_price": trigger.end_close,
        "entry_time": entry.end.isoformat(),
        "entry_sl": entry.sl,
        "entry_tp": entry.tp,
        "params": {
            "pullback_sr_tolerance": style.pullback_sr_tolerance,
            "pullback_max_days": style.pullback_max_days,
            "min_rr": style.entry_min_rr,
        },
        "context_pullback_tf": ctx_pullback,
        "context_trigger_tf": ctx_trigger,
    }

    system_prompt = """
Eres un trader profesional especializado en patrones de impulso + pullback + ruptura en criptomonedas.

MUY IMPORTANTE:
- El motor GREEN ya hizo el filtrado técnico fuerte de:
  - impulso
  - pullback
  - trigger
  - entry
- Tu tarea NO es buscar la perfección, sino detectar setups claramente malos o incoherentes.
- Si el setup es razonable, aunque no sea perfecto, DEBES aprobarlo (con la confianza que corresponda).

Ten en cuenta lo siguiente:

1) IMPULSO:
   - Asume que ya fue validado por el sistema.
   - Aunque impulse_start e impulse_end puedan ser iguales o parecidos en el payload, NO uses eso como motivo principal de rechazo.
   - Solo marca el impulso como problema si el contexto de precio muestra claramente un rango sin dirección o una estructura totalmente opuesta a la dirección del trade.

2) PULLBACK:
   - Debe retroceder hacia el nivel de soporte/resistencia sr_level.
   - Debe tener cierta estructura (no un simple "pincho" en una sola vela gigantesca).
   - Es válido que no sea una cuña perfecta.
   - Lo que buscamos penalizar es:
     - Pullbacks excesivamente verticales (muy rápidos, sin complejidad).
     - Pullbacks que superan claramente el sr_level y lo invalidan.

3) TRIGGER:
   - Debe aparecer cerca del final del pullback y en coherencia con la dirección del trade.
   - No necesitas ver una cuña diagonal perfecta; basta con que el trigger tenga sentido como ruptura de agotamiento del pullback.

4) ENTRY:
   - Debe estar relativamente cerca en el tiempo del trigger.
   - SL y TP deben tener sentido con el tamaño del pullback y el nivel sr_level.
   - Si el RR (rr) es muy bajo o el SL está claramente mal ubicado (ej. por encima de la entrada en un long), puedes rechazar.

CRITERIO DE DECISIÓN:

- Usa RECHAZO solo para setups claramente malos:
  - impulso totalmente plano o en contra.
  - pullback demasiado vertical sin estructura.
  - trigger muy lejos del final del pullback o en la dirección equivocada.
  - SL/TP absurdos (ej: SL del lado equivocado del trade).

- Si el setup es aceptable o simplemente “no estás seguro”, APRUÉBALO.
  - En esos casos, refleja la duda en el campo "confidence" (por ejemplo 0.5 o 0.6).

Debes devolver SIEMPRE un JSON EXACTO con este formato:

{
  "approved": true/false,
  "confidence": float entre 0 y 1,
  "reason": "texto corto explicando por qué"
}

No devuelvas nada más fuera de ese JSON.
"""

    user_prompt = (
        "Analiza este setup de trading dentro del contexto de impulso + pullback + trigger + entry "
        "que ya fueron filtrados por el motor GREEN.\n\n"
        "Recuerda:\n"
        "- Tu rol es ser un filtro SUAVE: solo rechaza setups claramente defectuosos.\n"
        "- Si el setup es razonable o solo leves dudas, apruébalo y usa un nivel de confidence acorde.\n"
        "- NO rechaces solo porque impulse_start e impulse_end sean iguales o porque la cuña no sea perfecta.\n\n"
        f"SETUP:\n```json\n{json.dumps(payload)}\n```"
    )

    resp = client.chat.completions.create(
        model=model,
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