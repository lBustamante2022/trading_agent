# app/green/styles.py
"""
GREEN v3 – Definición de estilos (DAY / SCALPING / SWING)

La idea es que *toda* la configuración específica de cada estilo
viva acá, en un único lugar, y el resto del sistema (pipeline,
backtest, live) solo lea estos valores.

Cada GreenStyle define:

    - Timeframes por etapa:
        impulse_tf
        pullback_tf
        trigger_tf
        entry_tf
        position_price_tf
        position_ema_tf

    - Parámetros de impulso (SR + ruptura)
    - Parámetros de pullback (cuña contra SR)
    - Parámetros de trigger (diagonal)
    - Parámetros de entry (lookahead)
    - Parámetros de gestión de riesgo y posición (SL, RR, EMA, vida máxima)

Más adelante, si querés, se puede agregar una capa para pisar estos
valores desde .ENV, pero la fuente de verdad sigue siendo este módulo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class GreenStyle:
    name: str

    # -----------------------------
    # Timeframes lógicos del pipeline
    # -----------------------------
    impulse_tf: str           # TF donde se detecta el impulso
    pullback_tf: str          # TF donde se mide el pullback
    trigger_tf: str           # TF donde se detecta el trigger
    entry_tf: str             # TF donde se toma la vela de entrada

    # Gestión de posición
    position_price_tf: str    # TF principal para simular la posición
    position_ema_tf: str      # TF donde se calcula la EMA de trailing
    max_holding_minutes: int  # Vida máxima de una operación

    # -----------------------------
    # Impulse / SR
    # -----------------------------
    impulse_min_body_pct: float        # mínimo tamaño de cuerpo vs precio
    impulse_break_pct: float           # % de ruptura mínima del SR
    impulse_sr_price_tol_pct: float    # tolerancia para agrupar swings en SR
    impulse_sr_min_touches: int        # toques mínimos para validar SR

    # -----------------------------
    # Pullback
    # -----------------------------
    pullback_sr_tolerance: float       # qué tan cerca del SR deben estar los swings
    pullback_min_hours: float          # duración mínima del pullback (horas)
    pullback_max_days: float           # ventana máxima tras el impulso (días)
    pullback_slow_body_factor: float

    # -----------------------------
    # Trigger
    # -----------------------------
    trigger_sr_tolerance: float        # tolerancia al SR para la diagonal
    trigger_lookback_minutes: int      # ventana de búsqueda hacia atrás
    trigger_max_break_minutes: int     # tiempo máximo entre ruptura y trigger
    trigger_entry_ema_length: float
    trigger_entry_body_min_ratio: float
    trigger_entry_ema_break_buffer_pct: float

    # -----------------------------
    # Entry
    # -----------------------------
    entry_max_minutes: int              # cuánto tiempo después del trigger buscamos la entrada
    entry_sl_buffer_pct: float               # buffer extra para el SL
    entry_min_rr: float                      # RR mínimo planeado para aceptar el trade

    # -----------------------------
    # Gestión de riesgo / posición
    # -----------------------------
    position_ema_trail_buffer_pct: float        # buffer de trailing respecto a la EMA
    position_ema_trail_lookback_bars: int       # barras usadas para “estabilizar” la EMA
    position_size: float
    position_ema_trail_span: int

# ----------------------------------------------------------------------
# Estilos predefinidos
# ----------------------------------------------------------------------

# === DAY TRADING (tu versión original adaptada a este modelo) ===
DAY_STYLE = GreenStyle(
    name="DAY",

    # Timeframes
    impulse_tf="4h",
    pullback_tf="1h",
    trigger_tf="15m",
    entry_tf="3m",
    position_price_tf="3m",
    position_ema_tf="5m",
   
    # Impulse / SR
    impulse_sr_price_tol_pct=0.01,  
    impulse_sr_min_touches=3,
    impulse_min_body_pct=0.05,     
    impulse_break_pct=0.01,         

    # Pullback
    pullback_sr_tolerance=0.005,   
    pullback_min_hours=24.0,       
    pullback_max_days=10.0,        
    pullback_slow_body_factor=0.6,

    # Trigger
    trigger_sr_tolerance=0.005,
    trigger_lookback_minutes=180,
    trigger_max_break_minutes=36 * 60,  # 36h
    trigger_entry_ema_length=50,
    trigger_entry_body_min_ratio=0.6,
    trigger_entry_ema_break_buffer_pct=0.0,

    # Entry
    entry_max_minutes=60,
    entry_sl_buffer_pct=0.001,
    entry_min_rr=2.0,

    # Gestión de riesgo / posición
    position_ema_trail_buffer_pct=0.002,    
    position_ema_trail_lookback_bars=50,
    position_size = 5.0,
    max_holding_minutes=10 * 24 * 60,  
    position_ema_trail_span=50
)


# === SCALPING ===
SCALPING_STYLE = GreenStyle(
    name="SCALPING",

    # Timeframes (tu propuesta)
    impulse_tf="15m",
    pullback_tf="5m",
    trigger_tf="1m",
    entry_tf="1m",
    position_price_tf="1m",
    position_ema_tf="1m",

    # Impulse / SR
    impulse_sr_price_tol_pct=0.005,  # 1% para agrupar SR en intradía
    impulse_sr_min_touches=3,
    impulse_min_body_pct=0.03,      # 8% de cuerpo para que realmente sea un “impulso”
    impulse_break_pct=0.015,         # ruptura un poco más fuerte (~1.5%)

    # Pullback
    pullback_sr_tolerance=0.005,    # un poco más cerca del SR
    pullback_min_hours=2,           # pullback mínimo ~2h
    pullback_max_days=1.0,          # ventana corta desde el impulso (intradía / 2 días)
    pullback_slow_body_factor=0.6,

    # Trigger
    trigger_sr_tolerance=0.01,
    trigger_lookback_minutes=180,
    trigger_max_break_minutes=120,   # 12h
    trigger_entry_ema_length=50,
    trigger_entry_body_min_ratio=0.6,
    trigger_entry_ema_break_buffer_pct=0.0,

    # Entry
    entry_max_minutes=60,
    entry_sl_buffer_pct=0.001,     # SL algo más ajustado
    entry_min_rr=1.5,               # algo más permisivo en scalping

    # Gestión de riesgo / posición
    position_ema_trail_buffer_pct=0.0015,
    position_ema_trail_lookback_bars=50,
    position_size = 5.0,
    max_holding_minutes=24 * 60,  
    position_ema_trail_span= 50   
)


# === SWING ===
SWING_STYLE = GreenStyle(
    name="SWING",

    # Timeframes (adaptado a marco mayor)
    impulse_tf="1w",    # impulso en semanal
    pullback_tf="1d",   # pullback en diario
    trigger_tf="1h",    # trigger en 1h
    entry_tf="15m",     # entrada en 15m

    position_price_tf="15m",
    position_ema_tf="30m",
    

    # Impulse / SR
    impulse_min_body_pct=0.02,
    impulse_break_pct=0.01,
    impulse_sr_price_tol_pct=0.0075,  # un poco más ancho (~0.75%)
    impulse_sr_min_touches=6,

    # Pullback
    pullback_sr_tolerance=0.025,
    pullback_min_hours=24.0,       # pullback mínimo ~1 día
    pullback_max_days=30.0,        # ventana más larga
    pullback_slow_body_factor=0.6,

    # Trigger
    trigger_sr_tolerance=0.05,
    trigger_lookback_minutes=24 * 60,     # 1 día hacia atrás
    trigger_max_break_minutes=7 * 24 * 60,  # hasta 7 días desde ruptura
    trigger_entry_ema_length=8,
    trigger_entry_body_min_ratio=0.6,
    trigger_entry_ema_break_buffer_pct=0.0,

    # Entry
    entry_max_minutes=4 * 60,    # hasta 4h después del trigger
    entry_sl_buffer_pct=0.0025,
    entry_min_rr=2.0,

    # Gestión de riesgo / posición
    position_ema_trail_buffer_pct=0.002,
    position_ema_trail_lookback_bars=50,
    position_size = 5.0,
    max_holding_minutes=7 * 24 * 60,  # 7 días
    position_ema_trail_span= 50
)


# ----------------------------------------------------------------------
# Registro y helper
# ----------------------------------------------------------------------

_STYLES_BY_NAME: Dict[str, GreenStyle] = {
    DAY_STYLE.name.upper(): DAY_STYLE,
    SCALPING_STYLE.name.upper(): SCALPING_STYLE,
    SWING_STYLE.name.upper(): SWING_STYLE,
}


def get_style(name: str) -> GreenStyle:
    """
    Devuelve el GreenStyle por nombre (case-insensitive).

    Ejemplos:
        get_style("DAY")
        get_style("scalping")
        get_style("swing")
    """
    key = name.strip().upper()
    if key not in _STYLES_BY_NAME:
        available = ", ".join(_STYLES_BY_NAME.keys())
        raise ValueError(f"Estilo GREEN desconocido: {name!r}. Disponibles: {available}")
    return _STYLES_BY_NAME[key]