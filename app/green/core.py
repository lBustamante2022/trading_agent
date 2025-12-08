# app/green/core.py
"""
GREEN v3 – Núcleo orquestador (GreenV3Core)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Dict, Any, Optional
from datetime import datetime, timedelta

import builtins as _builtins
import pandas as pd

from app.green.styles import GreenStyle
from app.exchange.base import IExchange

DEBUG_CORE = False

def _log(*args, **kwargs):
    if DEBUG_CORE:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = " ".join(str(a) for a in args)
        _builtins.print(f"[{ts} DEBUG CORE] {msg}", **kwargs)

print = _log  # override local

# ============================================================
# Tipos de dominio (value objects)
# ============================================================

@dataclass(frozen=True)
class Impulse:
    symbol: str
    direction: str              # "long" / "short"
    start: pd.Timestamp
    end: pd.Timestamp
    sr_level: float


@dataclass(frozen=True)
class Pullback:
    symbol: str
    direction: str
    impulse: Impulse
    start: pd.Timestamp
    end: pd.Timestamp
    # swings: List[Dict[str, Any]]
    end_close: float                 


@dataclass(frozen=True)
class Trigger:
    symbol: str
    direction: str
    pullback: Pullback
    start: pd.Timestamp
    end: pd.Timestamp
    end_close: float


@dataclass(frozen=True)
class Entry:
    symbol: str
    direction: str
    trigger: Trigger
    start: pd.Timestamp
    end: pd.Timestamp
    entry: float
    sl: float
    tp: float


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
        impulse: Impulse,
        style: GreenStyle,
        cfg: Any,
    ) -> List[Pullback]:
        ...


class TriggerStrategy(Protocol):
    def detect_triggers(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        pullback: Pullback,
        style: GreenStyle,
        cfg: Any,
    ) -> List[Trigger]:
        ...


class EntryStrategy(Protocol):
    def detect_entry(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        trigger: Trigger,
        style: GreenStyle,
        cfg: Any,
    ) -> Optional[Entry]:
        ...


class PositionStrategy(Protocol):
    def simulate_trade(
        self,
        symbol: str,
        dfs: Dict[str, pd.DataFrame],
        entry: Entry,
        style: GreenStyle,
        cfg: Any,
    ) -> Optional[TradeResult]:
        ...


# ============================================================
# Conjunto de estrategias
# ============================================================

@dataclass(frozen=True)
class GreenStages:
    impulse: ImpulseStrategy
    pullback: PullbackStrategy
    trigger: TriggerStrategy
    entry: EntryStrategy
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
        cfg: Any
    ) -> List[TradeResult]:
        trades: List[TradeResult] = []

        impulses = self.stages.impulse.detect_impulses(
            symbol=symbol,
            dfs=dfs,
            style=self.style,
            cfg=cfg,
        )
        if not impulses:
            return trades

        print(f"impulses={len(impulses)}")
        for imp in impulses:
            print(f"IMP {imp.direction} start={imp.start} SR={imp.sr_level:.4f} ")
            took_direction = {"long": False, "short": False}

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
                print(f"PBK {pb.direction} start={pb.start} price={pb.end_close} ")

                triggers = self.stages.trigger.detect_triggers(
                    symbol=symbol,
                    dfs=dfs,
                    pullback=pb,
                    style=self.style,
                    cfg=cfg,
                )
                if not triggers:
                    continue

                for tg in triggers:
                    print(f"TGR {pb.direction} price={pb.end_close:.4f} start={pb.start} end={pb.end} ")
                    if took_direction.get(tg.direction, False):
                        continue

                    entry = self.stages.entry.detect_entry(
                        symbol=symbol,
                        dfs=dfs,
                        trigger=tg,
                        style=self.style,
                        cfg=cfg,
                    )
                    if entry is None:
                        continue

                    print(f"ETR {entry.direction} entry={entry.entry} tp={entry.tp} sl={entry.sl} time={entry.end}")
                    trade_res = self.stages.position.simulate_trade(
                        symbol=symbol,
                        dfs=dfs,
                        entry=entry,
                        style=self.style,
                        cfg=cfg,
                    )
                    if trade_res is None:
                        continue

                    trades.append(trade_res)
                    took_direction[trade_res.direction] = True
                    break

        trades.sort(key=lambda t: t.entry_time)
        print(f"trades={len(trades)}")
        return trades