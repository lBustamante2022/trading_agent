# app/green/core.py
"""
GREEN v3 – Núcleo orquestador (GreenV3Core)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Dict, Any, Optional
import builtins as _builtins
import pandas as pd

from app.green.styles import GreenStyle
from app.exchange.base import IExchange  # puede seguir usándose en PositionStrategy

DEBUG_CORE = False


def _log(*args, **kwargs):
    if DEBUG_CORE:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = " ".join(str(a) for a in args)
        _builtins.print(f"[{ts} DEBUG CORE] {msg}", **kwargs)


print = _log  # override local

# ============================================================
# Tipos de dominio (value objects)
# ============================================================


@dataclass(frozen=True)
class Impulse:
    meta: Dict[str, Any]
    # Ejemplos de meta:
    #   "symbol": str
    #   "direction": "long" | "short"
    #   "start": pd.Timestamp
    #   "end": pd.Timestamp
    #   "sr_level": float
    #   "sr_kind": "support" | "resistance"
    #   ...


@dataclass(frozen=True)
class Pullback:
    impulse: Impulse
    meta: Dict[str, Any]
    # Ejemplos de meta:
    #   "symbol": str
    #   "direction": "long" | "short"
    #   "start": pd.Timestamp
    #   "end": pd.Timestamp
    #   "end_close": float
    #   "sr_level": float
    #   "sr_kind": ...
    #   "sr_touch_ts": ...
    #   "rebound_ts": ...
    #   ...


@dataclass(frozen=True)
class Trigger:
    pullback: Pullback
    meta: Dict[str, Any]
    # No se usa en el pipeline nuevo, pero se mantiene
    # por compatibilidad con código viejo.


@dataclass(frozen=True)
class Entry:
    trigger: Trigger
    meta: Dict[str, Any]
    # meta:
    #   "symbol": str
    #   "direction": "long" | "short"
    #   "start": pd.Timestamp
    #   "end": pd.Timestamp        # tiempo de la vela de entrada
    #   "entry": float
    #   "sl": float
    #   "tp": float
    #   "planned_rr": float
    #   ...


@dataclass(frozen=True)
class TradeResult:
    symbol: str
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    sl_initial: float
    tp_initial: float
    exit_time: pd.Timestamp
    exit_price: float
    result: str          # "win" / "loss" / "be"
    rr_planned: float
    rr_real: float
    trigger_time: pd.Timestamp
    impulse_start: pd.Timestamp


# ============================================================
# Interfaces de estrategias
# ============================================================

class ImpulseStrategy(Protocol):
    def detect_impulses(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
    ) -> List[Impulse]:
        ...


class PullbackStrategy(Protocol):
    def detect_pullbacks(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        impulse: Impulse,
    ) -> List[Pullback]:
        ...


# Se mantienen por compatibilidad (no usados en el pipeline nuevo)
class TriggerStrategy(Protocol):
    def detect_triggers(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        pullback: Pullback,
    ) -> List[Trigger]:
        ...


class EntryStrategy(Protocol):
    def detect_entry(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        trigger: Trigger,
    ) -> Optional[Entry]:
        ...


class PositionStrategy(Protocol):
    def gestion_trade(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        entry: Entry,
    ) -> Optional[TradeResult]:
        ...


class TriggerAIStrategy(Protocol):
    def decide_entry(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        style: GreenStyle,
        cfg: Any,
        pullback: Pullback,
    ) -> Optional[Entry]:
        """
        Devuelve un Entry completo (entry, SL, TP, RR, tiempos, etc.)
        o None si la lógica / IA decide NO operar ese pullback.
        """
        ...


# ============================================================
# Conjunto de estrategias
# ============================================================


@dataclass(frozen=True)
class GreenStages:
    impulse: ImpulseStrategy
    pullback: PullbackStrategy
    trigger_ai: TriggerAIStrategy
    position: PositionStrategy


# ============================================================
# Núcleo orquestador
# ============================================================


@dataclass
class GreenV3Core:
    style: GreenStyle
    stages: GreenStages

    def run(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        cfg: Any,
    ) -> List[TradeResult]:
        """
        Orquesta todo el flujo:

            IMPULSO → PULLBACK → TRIGGER_AI → POSITION

        Devuelve la lista de TradeResult (trades simulados / ejecutados).

        NOTA importante:
        - Para evitar "overfitting de backtest", si hay un trade abierto
          (trade_timestamp_end) se descartan los impulsos que caen
          dentro de la vida de ese trade. En live, durante una posición
          abierta, no operaríamos nuevos impulsos en paralelo sobre
          el mismo símbolo/estrategia.
        """
        # -------------------------------
        # 1) IMPULSOS
        # -------------------------------
        impulses = self.stages.impulse.detect_impulses(
            symbol=symbol,
            dfs=dfs,
            style=self.style,
            cfg=cfg,
        )
        if not impulses:
            print("impulses=0")
            return []

        print(f"impulses={len(impulses)}")
        for imp in impulses:
            print(
                f"IMP {imp.meta['direction']} "
                f"start={imp.meta['start']} SR={imp.meta['sr_level']:.4f}"
            )

        # -------------------------------
        # 2) LOOP IMPULSO → PULLBACK → ENTRY → POSITION
        # -------------------------------
        trades: List[TradeResult] = []
        trade_timestamp_end: Optional[pd.Timestamp] = None

        for imp in impulses:
            # Si todavía tenemos un trade activo (en backtest),
            # descartamos impulsos cuyo fin sea anterior al fin de ese trade.
            # Esto evita que se apilen impulsos dentro de la vida del trade.
            if trade_timestamp_end is not None and imp.meta["end"] < trade_timestamp_end:
                continue

            took_direction = {"long": False, "short": False}

            # 2.a) PULLBACKS para este impulso
            pullbacks = self.stages.pullback.detect_pullbacks(
                symbol=symbol,
                dfs=dfs,
                impulse=imp,
                style=self.style,
                cfg=cfg,
            )
            if not pullbacks:
                continue

            for pb in pullbacks:
                print(
                    f"PBK {pb.meta['direction']} "
                    f"start={pb.meta['start']} price={pb.meta.get('end_close')}"
                )

                # 2.b) TRIGGER_AI → decide entry o no
                entry = self.stages.trigger_ai.decide_entry(
                    symbol=symbol,
                    dfs=dfs,
                    style=self.style,
                    cfg=cfg,
                    pullback=pb,
                )
                if entry is None:
                    continue

                print(
                    f"ETR {entry.meta['direction']} "
                    f"entry={entry.meta['entry']} "
                    f"tp={entry.meta['tp']} sl={entry.meta['sl']} "
                    f"time={entry.meta['end']}"
                )

                # No permitir dos trades de la MISMA dirección para el mismo impulso
                if took_direction.get(entry.meta["direction"], False):
                    continue

                # 2.c) POSITION → gestiona el trade hasta SL / TP / timeout
                trade_res = self.stages.position.gestion_trade(
                    symbol=symbol,
                    dfs=dfs,
                    entry=entry,
                    style=self.style,
                    cfg=cfg,
                )
                if trade_res is None:
                    continue

                # Marcar fin de trade para evitar nuevos impulsos dentro
                trade_timestamp_end = trade_res.exit_time

                trades.append(trade_res)
                took_direction[trade_res.direction] = True

                # Para este impulso, ya tomamos una operación en esta dirección;
                # salimos del loop de pullbacks.
                break

        trades.sort(key=lambda t: t.entry_time)
        print(f"trades={len(trades)}")
        return trades