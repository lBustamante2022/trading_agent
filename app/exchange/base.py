# app/exchange/base.py
"""
Interfaces base para el módulo de Exchange en GREEN v3.

La idea es desacoplar completamente la lógica de GREEN (impulso, pullback,
trigger, entry, position) de la fuente real de datos / órdenes:

    - BacktestExchange  -> usa data histórica en memoria (CSV)
    - OkxExchange       -> usa API real de OKX

Cualquier implementación concreta debe cumplir la interfaz IExchange.
"""

from __future__ import annotations

from typing import Protocol, Optional, Dict, Any, Iterator
from datetime import datetime

import pandas as pd


class IExchange(Protocol):
    """
    Contrato mínimo que GREEN v3 necesita para interactuar con un "exchange"
    (real o simulado).

    NOTA: acá NO hay ninguna lógica de GREEN ni de estilos; solamente
    operaciones genéricas sobre datos de mercado y posiciones.
    """

    # ------------------------
    # Datos de mercado
    # ------------------------
    def fetch_candles(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Devuelve un DataFrame con velas OHLCV para el símbolo y timeframe dado.

        Columnas esperadas:
            - timestamp (pd.Timestamp)
            - open, high, low, close (float)
            - volume (opcional)

        Si `start` y/o `end` se especifican, se debe filtrar por rango de tiempo.
        Si no hay datos, devolver DataFrame vacío.
        """
        ...

    # ------------------------
    # Gestión de posiciones
    # ------------------------
    def create_position(
        self,
        symbol: str,
        side: str,          # "long" o "short"
        size: float,
        entry_price: float,
        sl: float,
        tp: float,
    ) -> str:
        """
        Crea una nueva posición:

            - En backtest: posición "virtual" en memoria.
            - En live: orden real (p.ej. market + SL/TP en OKX).

        Debe devolver un identificador único de posición (position_id),
        que luego se usará para actualizar SL/TP o cerrar la posición.
        """
        ...

    def update_sl_tp(
        self,
        position_id: str,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> None:
        """
        Actualiza SL y/o TP de una posición existente.

            - En backtest: actualiza el estado interno en memoria.
            - En live: envía las modificaciones correspondientes al broker.

        Los parámetros que vengan como None no deben modificarse.
        """
        ...

    def close_position(
        self,
        position_id: str,
        reason: str,
    ) -> None:
        """
        Cierra una posición:

            - En backtest: marca la posición como cerrada / la remueve del store.
            - En live: envía la orden de cierre (market o equivalente).

        `reason` es solo texto informativo (p.ej. "timeout", "manual", etc.).
        """
        ...

    # ------------------------
    # Tiempo de referencia
    # ------------------------
    def now(self) -> datetime:
        """
        Devuelve la "hora actual" para el contexto:

            - En backtest: puede ser el timestamp de la vela que se está
              procesando o un reloj simulado.
            - En live: normalmente datetime.utcnow().

        Sirve para logs, timeouts, y decisiones dependientes del tiempo.
        """
        ...

    
    def iter_position_bars(
        self,
        symbol: str,
        price_tf: str,
        ema_tf: str,
        start_time: pd.Timestamp,
        max_minutes: int,
        ema_span: int,
    ) -> Iterator[Dict[str, Any]]:
        """
        Iterador de barras para gestión de posición.

        Debe rendir dicts con:
            - "timestamp": pd.Timestamp
            - "high": float
            - "low": float
            - "close": float
            - "ema": float  (EMA calculada en ema_tf con span=ema_span)

        Position NO debe saber si está en backtest o live.
        BacktestExchange lo implementa leyendo de DF internos;
        OkxExchange lo implementa consultando al exchange real.
        """
        ...