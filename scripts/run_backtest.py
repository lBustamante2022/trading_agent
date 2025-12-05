# scripts/run_backtest.py

import os
import argparse
from dataclasses import asdict
from typing import Dict

import pandas as pd
import matplotlib.pyplot as plt

from app.green.core import GreenV3Core, GreenStages
from app.green.styles import get_style, GreenStyle
from app.green.pipeline.impulse import DefaultImpulseStrategy
from app.green.pipeline.pullback import DefaultPullbackStrategy
from app.green.pipeline.trigger import DefaultTriggerStrategy
from app.green.pipeline.entry import DefaultEntryStrategy
from app.green.pipeline.position import DefaultPositionStrategy
from app.exchange.backtest import BacktestExchange
from app.strategies.green.types import GreenConfig   # reutilizamos config vieja


# ============================================================
# Loader genérico de velas
# ============================================================

def _load_tf_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_timeframes_for_symbol(
    symbol: str,
    base_dir: str,
) -> Dict[str, pd.DataFrame]:
    os.makedirs(base_dir, exist_ok=True)

    sym_clean = symbol.replace("/", "")
    timeframes = ["1m", "3m", "5m", "15m", "1h", "4h", "1d", "1w"]
    dfs: Dict[str, pd.DataFrame] = {}

    for tf in timeframes:
        fname = f"binance_{sym_clean}_{tf}.csv"
        fpath = os.path.join(base_dir, fname)
        df = _load_tf_csv(fpath)
        dfs[tf] = df

    return dfs


# ============================================================
# Helper: recortar últimos N días
# ============================================================

def filter_last_days(dfs: Dict[str, pd.DataFrame], days: int) -> Dict[str, pd.DataFrame]:
    base_tf = None
    for tf in ("1m", "3m", "5m", "15m", "1h", "4h", "1d", "1w"):
        df = dfs.get(tf)
        if df is not None and not df.empty and "timestamp" in df.columns:
            base_tf = df
            break

    if base_tf is None:
        return dfs

    last_ts = base_tf["timestamp"].max()
    cutoff = last_ts - pd.Timedelta(days=days)

    out: Dict[str, pd.DataFrame] = {}
    for tf, df in dfs.items():
        if df is None or df.empty or "timestamp" not in df.columns:
            out[tf] = df
            continue
        out[tf] = df[df["timestamp"] >= cutoff].reset_index(drop=True)

    return out


# ============================================================
# Gráficos: Equity + Histograma
# ============================================================

def plot_equity_and_hist(trades_df: pd.DataFrame, symbol: str, style: GreenStyle, output_dir: str):
    if trades_df is None or trades_df.empty:
        print(f"[{symbol}] No hay trades para graficar.")
        return

    trades_df = trades_df.sort_values("entry_time").reset_index(drop=True)

    rr_col = "rr_real" if "rr_real" in trades_df.columns else "rr_planned"
    if rr_col not in trades_df.columns:
        print(f"[{symbol}] WARNING: no encontré rr_real ni rr_planned, no puedo graficar equity.")
        return

    trades_df["cum_rr"] = trades_df[rr_col].cumsum()

    os.makedirs(output_dir, exist_ok=True)
    sym_clean = symbol.replace("/", "")

    plt.figure(figsize=(10, 4))
    plt.plot(trades_df["cum_rr"])
    plt.title(f"Equity Curve - {symbol} ({style.name})")
    plt.xlabel("Trade #")
    plt.ylabel("Cumulative R")
    plt.grid(True)
    path1 = os.path.join(output_dir, f"equity_{sym_clean}_{style.name}.png")
    plt.savefig(path1, bbox_inches="tight")
    plt.close()
    print(f"[{symbol}] Equity curve guardada en {path1}")

    plt.figure(figsize=(7, 4))
    plt.hist(trades_df[rr_col], bins=40)
    plt.title(f"RR Histogram - {symbol} ({style.name})")
    plt.xlabel("R")
    plt.ylabel("Count")
    plt.grid(True)
    path2 = os.path.join(output_dir, f"rr_hist_{sym_clean}_{style.name}.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"[{symbol}] Histograma guardado en {path2}")


# ============================================================
# Backtest runner
# ============================================================

def run_backtest(
    symbol: str,
    style: GreenStyle,
    cfg: GreenConfig,
    data_dir: str,
    output_dir: str,
    days: int = 360,
):
    print(f"\n========== BACKTEST {symbol} ({style.name}) ==========")

    dfs = load_timeframes_for_symbol(symbol, base_dir=data_dir)
    if not dfs or all((df is None or df.empty) for df in dfs.values()):
        print(f"[{symbol}] ERROR: No se pudieron cargar velas desde {data_dir}.")
        return None

    print(f"[{symbol}] Recortando últimos {days} días...")
    dfs = filter_last_days(dfs, days=days)

    impulse_strategy = DefaultImpulseStrategy()
    pullback_strategy = DefaultPullbackStrategy()
    trigger_strategy = DefaultTriggerStrategy()
    entry_strategy = DefaultEntryStrategy()
    position_strategy = DefaultPositionStrategy()

    stages = GreenStages(
        impulse=impulse_strategy,
        pullback=pullback_strategy,
        trigger=trigger_strategy,
        entry=entry_strategy,
        position=position_strategy,
    )

    exchange = BacktestExchange()

    core = GreenV3Core(
        style=style,
        stages=stages,
    )

    print(f"[{symbol}] Ejecutando GreenV3Core.run()...")
    trades = core.run(
        symbol=symbol,
        dfs=dfs,
        cfg=cfg,
        exchange=exchange,
    )

    if not trades:
        print(f"[{symbol}] No se generaron trades.")
        return None

    print(f"[{symbol}] Backtest finalizado, trades={len(trades)}")

    df = pd.DataFrame([asdict(t) for t in trades])

    os.makedirs(output_dir, exist_ok=True)
    sym_clean = symbol.replace("/", "")
    out_csv = os.path.join(output_dir, f"backtest_{sym_clean}_{style.name}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[{symbol}] Trades guardados en {out_csv} ({len(df)} filas).")

    plot_equity_and_hist(df, symbol, style, output_dir)

    return df


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest GREEN V3 (arquitectura nueva)")
    parser.add_argument("--symbol", type=str, required=True, help="Ej: BTC/USDT")
    parser.add_argument("--style", type=str, default="DAY", help="DAY / SCALPING / SWING")
    parser.add_argument("--data-dir", type=str, default="input", help="Carpeta de velas (CSV)")
    parser.add_argument("--output-dir", type=str, default="output", help="Carpeta de resultados")
    parser.add_argument("--days", type=int, default=360, help="Días hacia atrás a considerar")

    args = parser.parse_args()

    symbol = args.symbol
    style = get_style(args.style)

    cfg = GreenConfig(
        channel_width_pct=5.0,
        pullback_sr_tolerance=0.02,
        pullback_min_swings=2,
        pullback_max_days=10.0,
        trigger_min_swings=2,
        trigger_sr_tolerance=0.05,
        trigger_lookback_minutes=180,
        trigger_max_break_minutes=2160,
        entry_lookahead_minutes=60,
        sl_buffer_pct=0.0015,
        min_rr=2.0,
        ema_trail_buffer_pct=0.002,
        ema_trail_span=50,
         ai_supervisor_enabled=True,   # <-- activa IA
    )

    run_backtest(
        symbol=symbol,
        style=style,
        cfg=cfg,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        days=args.days,
    )


if __name__ == "__main__":
    main()