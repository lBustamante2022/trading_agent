# scripts/run_backtest.py

import os
import argparse
from dataclasses import asdict, replace
from typing import Dict, Any, List
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
import app.ta.environment

from app.green.core import GreenV3Core, GreenStages
from app.green.styles import get_style, GreenStyle
from app.green.pipeline.impulse import DefaultImpulseStrategy
from app.green.pipeline.pullback import DefaultPullbackStrategy
from app.green.pipeline.trigger import DefaultTriggerStrategy
from app.green.pipeline.entry import DefaultEntryStrategy
from app.green.pipeline.position import DefaultPositionStrategy
from app.exchange.backtest import BacktestExchange


# ============================================================
# Loader genérico de velas
# ============================================================

def _load_tf_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "timestamp" in df.columns:
        # Si viene como string numérico, lo pasamos a int primero
        # (por si hay cosas tipo "1673481600000")
        if df["timestamp"].dtype == object:
            try:
                df["timestamp"] = df["timestamp"].astype("int64")
            except Exception:
                # Si falla, que intente parsear como datetime normal
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                # Ahora sí: epoch en milisegundos → datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
            # Si ya es numérico, asumimos ms también
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

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

def iter_param_grid(grid: Dict[str, List[Any]]):
    keys = list(grid.keys())
    values_list = [grid[k] for k in keys]
    for values in product(*values_list):
        yield {k: v for k, v in zip(keys, values)}

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
    cfg: Any,
    input_dir: str,
    output_dir: str,
    days: int = 360,
    ai_supervisor: bool = False
):
    print(f"\n========== BACKTEST {symbol} ({style.name}) ==========")

    dfs = load_timeframes_for_symbol(symbol, base_dir=input_dir)
    if not dfs or all((df is None or df.empty) for df in dfs.values()):
        print(f"[{symbol}] ERROR: No se pudieron cargar velas desde {input_dir}.")
        return None

    print(f"[{symbol}] Recortando últimos {days} días...")
    dfs = filter_last_days(dfs, days=days)

    exchange = BacktestExchange()
    impulse_strategy = DefaultImpulseStrategy()
    pullback_strategy = DefaultPullbackStrategy()
    trigger_strategy = DefaultTriggerStrategy()
    entry_strategy = DefaultEntryStrategy(ai_supervisor=ai_supervisor)
    position_strategy = DefaultPositionStrategy(exchange=exchange)

    stages = GreenStages(
        impulse=impulse_strategy,
        pullback=pullback_strategy,
        trigger=trigger_strategy,
        entry=entry_strategy,
        position=position_strategy,
    )

    core = GreenV3Core(
        style=style,
        stages=stages,
    )

    print(f"[{symbol}] Ejecutando GreenV3Core.run()...")
    trades = core.run(
        symbol=symbol,
        dfs=dfs,
        cfg=cfg
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
# OPTIMIZER para SCALPING
# ============================================================

def run_optimizer(
    symbol: str,
    base_style: GreenStyle,
    input_dir: str,
    output_dir: str,
    days: int = 360,
    ai_supervisor: bool = False,
):
    """
    Ejecuta un grid de parámetros basado en SCALPING_STYLE y
    genera un CSV con:
      - symbol
      - parámetros de la configuración
      - resultados (wins, be, losses, total_R, avg_R)
    """

    if base_style.name.upper() != "SCALPING":
        print(f"[{symbol}] WARNING: el optimizador está pensado para SCALPING, estilo actual={base_style.name}")
    
    # --- Definí acá el grid a probar (puedes ajustarlo a gusto) ---
    param_grid: Dict[str, List[Any]] = {
        "impulse_min_body_pct":   [0.02, 0.03, 0.04],
        "impulse_break_pct":      [0.003, 0.005, 0.007],

        "pullback_sr_tolerance":  [0.003, 0.005, 0.007],
        "pullback_slow_body_factor": [0.5, 0.6, 0.7],
    
        "trigger_sr_tolerance":   [0.003, 0.005, 0.007],

        "entry_sl_buffer_pct":    [0.0005, 0.0010],
        "entry_min_rr":           [1.2, 1.5, 2.0],
    }

    results_rows: List[Dict[str, Any]] = []

    for params in iter_param_grid(param_grid):
        print("\n===============================================")
        print(f"[{symbol}] Probando config: {params}")
        print("===============================================")

        # Creamos un nuevo estilo con esos parámetros pisados
        style_cfg = replace(base_style, **params)

        # Reutilizamos run_backtest (usa style_cfg como style y cfg)
        trades_df = run_backtest(
            symbol=symbol,
            style=style_cfg,
            cfg=style_cfg,
            input_dir=input_dir,
            output_dir=output_dir,
            days=days,
            ai_supervisor=ai_supervisor,
        )

        if trades_df is None or trades_df.empty:
            print(f"[{symbol}] Sin trades para esta config, la salteo.")
            row = {
                "symbol": symbol,
                **params,
                "trades": 0,
                "wins": 0,
                "be": 0,
                "losses": 0,
                "total_R": 0.0,
                "avg_R": 0.0,
            }
            results_rows.append(row)
            continue

        # Determinar qué columna de R usar
        rr_col = "rr_real" if "rr_real" in trades_df.columns else "rr_planned"
        if rr_col not in trades_df.columns:
            print(f"[{symbol}] WARNING: no encontré rr_real ni rr_planned, no puedo calcular stats.")
            continue

        wins = int((trades_df[rr_col] > 0).sum())
        be = int((trades_df[rr_col] == 0).sum())
        losses = int((trades_df[rr_col] < 0).sum())
        total_R = float(trades_df[rr_col].sum())
        avg_R = float(trades_df[rr_col].mean())

        row = {
            "symbol": symbol,
            **params,
            "trades": len(trades_df),
            "wins": wins,
            "be": be,
            "losses": losses,
            "total_R": total_R,
            "avg_R": avg_R,
        }
        results_rows.append(row)

    if not results_rows:
        print(f"[{symbol}] No se generaron resultados en el optimizer.")
        return

    results_df = pd.DataFrame(results_rows)
    os.makedirs(output_dir, exist_ok=True)
    sym_clean = symbol.replace("/", "")
    out_csv = os.path.join(output_dir, f"optimize_{sym_clean}_{base_style.name}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"[{symbol}] Resultados de optimización guardados en {out_csv} ({len(results_df)} filas).")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Backtest GREEN V3 (arquitectura nueva)")
    parser.add_argument("--symbol", type=str, required=True, help="Ej: BTC/USDT")
    parser.add_argument("--style", type=str, default="DAY", help="DAY / SCALPING / SWING")
    parser.add_argument("--input-dir", type=str, default="input", help="Carpeta de velas (CSV)")
    parser.add_argument("--output-dir", type=str, default="output", help="Carpeta de resultados")
    parser.add_argument("--days", type=int, default=360, help="Días hacia atrás a considerar")
    parser.add_argument("--ai", action="store_true", help="Usar supervisor IA para validar setups antes de simular el trade")
    parser.add_argument("--optimize", action="store_true", help="Ejecutar grid search de parámetros (SCALPING optimizer)")
    args = parser.parse_args()

    symbol = args.symbol
    style = get_style(args.style)
    cfg = style
    use_ai = args.ai

    if args.optimize:
        # Modo OPTIMIZER
        run_optimizer(
            symbol=symbol,
            base_style=style,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            days=args.days,
            ai_supervisor=use_ai,
        )
    else:
        # Modo backtest normal (una sola config)
        run_backtest(
            symbol=symbol,
            style=style,
            cfg=cfg,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            days=args.days,
            ai_supervisor=use_ai,
        )

if __name__ == "__main__":
    main()