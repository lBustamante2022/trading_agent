# scripts/run_backtest.py

import os
import argparse
from dataclasses import asdict, replace
from typing import Dict, Any, List
from itertools import product
from datetime import timedelta

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle

import app.ta.environment

from app.green.core import GreenV3Core, GreenStages
from app.green.styles import get_style, GreenStyle
from app.green.pipeline.impulse import DefaultImpulseStrategy
from app.green.pipeline.pullback import DefaultPullbackStrategy
from app.green.pipeline.trigger_ai import DefaultTriggerAIStrategy
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
        if df["timestamp"].dtype == object:
            try:
                df["timestamp"] = df["timestamp"].astype("int64")
            except Exception:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            else:
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        else:
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


def _choose_plot_df(dfs: Dict[str, pd.DataFrame], preferred_tf: str) -> pd.DataFrame:
    """
    Intenta usar el TF preferido; si no hay datos, prueba otros estándares.
    """
    df = dfs.get(preferred_tf)
    if df is not None and not df.empty and "timestamp" in df.columns and "close" in df.columns:
        return df.sort_values("timestamp").reset_index(drop=True)

    for tf in ["15m", "5m", "1h", "1d", "3m", "1m", "4h", "1w"]:
        df = dfs.get(tf)
        if df is not None and not df.empty and "timestamp" in df.columns and "close" in df.columns:
            return df.sort_values("timestamp").reset_index(drop=True)

    return pd.DataFrame()


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
# Dibujo de trades (rectángulos riesgo / reward)
# ============================================================

def draw_trades_on_ax(ax, trades_df: pd.DataFrame):
    """
    Dibuja:
      - Entry (marker según dirección)
      - Exit (X)
      - Línea entry→exit
      - SL y TP originales como líneas
      - Rectángulos: rojo (riesgo) y verde (reward)
    """
    if trades_df is None or trades_df.empty:
        return

    for col in ["entry_time", "exit_time"]:
        if not np.issubdtype(trades_df[col].dtype, np.datetime64):
            trades_df[col] = pd.to_datetime(trades_df[col])

    for _, tr in trades_df.iterrows():
        direction = tr["direction"]
        result = tr["result"]

        entry_t = pd.to_datetime(tr["entry_time"])
        exit_t = pd.to_datetime(tr["exit_time"])
        entry_p = float(tr["entry_price"])
        exit_p = float(tr["exit_price"])
        sl = float(tr["sl_initial"])
        tp = float(tr["tp_initial"])
        rr_real = float(tr.get("rr_real", np.nan))

        if result == "win":
            color = "green"
        elif result == "loss":
            color = "red"
        else:
            color = "gray"

        entry_marker = "^" if direction == "long" else "v"

        # Entry y exit
        ax.scatter(entry_t, entry_p, marker=entry_marker, color=color,
                   s=60, zorder=5, label="_nolegend_")
        ax.scatter(exit_t, exit_p, marker="x", color=color,
                   s=60, zorder=5, label="_nolegend_")

        # Línea entry → exit
        ax.plot([entry_t, exit_t], [entry_p, exit_p],
                linestyle="--", color=color, alpha=0.7, zorder=4,
                label="_nolegend_")

        # SL / TP líneas horizontales
        ax.hlines(sl, xmin=entry_t, xmax=exit_t,
                  colors=color, linestyles=":", alpha=0.3, label="_nolegend_")
        ax.hlines(tp, xmin=entry_t, xmax=exit_t,
                  colors=color, linestyles=":", alpha=0.3, label="_nolegend_")

        # Rectángulos riesgo (rojo) y reward (verde)
        x_entry = mdates.date2num(entry_t)
        x_exit = mdates.date2num(exit_t)
        width = x_exit - x_entry
        if width <= 0:
            width = 1.0 / (24 * 60)  # fallback 1 minuto

        if direction == "long":
            risk_y = sl
            risk_h = max(entry_p - sl, 0)
            reward_y = entry_p
            reward_h = max(tp - entry_p, 0)
        else:
            risk_y = entry_p
            risk_h = max(sl - entry_p, 0)
            reward_y = tp
            reward_h = max(entry_p - tp, 0)

        if risk_h > 0:
            rect_risk = Rectangle(
                (x_entry, risk_y),
                width,
                risk_h,
                facecolor="red",
                edgecolor="red",
                alpha=0.1,
                linewidth=1.0,
                zorder=2,
            )
            ax.add_patch(rect_risk)

        if reward_h > 0:
            rect_reward = Rectangle(
                (x_entry, reward_y),
                width,
                reward_h,
                facecolor="green",
                edgecolor="green",
                alpha=0.1,
                linewidth=1.0,
                zorder=2,
            )
            ax.add_patch(rect_reward)

        txt = f"{result.upper()} {rr_real:.1f}R" if not np.isnan(rr_real) else result.upper()
        ax.text(
            entry_t,
            entry_p,
            txt,
            fontsize=8,
            color=color,
            ha="left",
            va="bottom",
        )


# ============================================================
# Construcción de pullbacks para gráficos
# ============================================================

def build_pullbacks_dataframe(
    symbol: str,
    dfs: Dict[str, pd.DataFrame],
    style: GreenStyle,
    impulse_strategy: DefaultImpulseStrategy,
    pullback_strategy: DefaultPullbackStrategy,
) -> pd.DataFrame:
    """
    Reconstruye Impulses + Pullbacks usando directamente las estrategias.
    No interviene el core ni la lógica de trades.
    """
    impulses = impulse_strategy.detect_impulses(
        symbol=symbol,
        dfs=dfs,
        style=style,
        cfg=style,
    )
    if not impulses:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []

    for imp in impulses:
        pbs = pullback_strategy.detect_pullbacks(
            symbol=symbol,
            dfs=dfs,
            impulse=imp,
            style=style,
            cfg=style,
        )
        if not pbs:
            continue

        for pb in pbs:
            meta = pb.meta
            row = {
                "symbol": symbol,
                "direction": meta.get("direction"),
                "pb_start": meta.get("start"),
                "pb_end": meta.get("end"),
                "pb_end_close": meta.get("end_close"),
                "sr_level": meta.get("sr_level", imp.meta.get("sr_level")),
                "sr_kind": meta.get("sr_kind", imp.meta.get("sr_kind", None)),
                "impulse_start": imp.meta.get("start"),
                "impulse_end": imp.meta.get("end"),
                "impulse_direction": imp.meta.get("direction"),
                "impulse_sr_level": imp.meta.get("sr_level"),
            }
            for k in [
                "sr_touch_ts",
                "sr_touch_price",
                "sr_touch_dist_pct",
                "avg_body_impulse_pct",
                "avg_body_pullback_pct",
                "rebound_body_ratio",
                "rebound_ts",
                "rebound_dir",
            ]:
                if k in meta:
                    row[k] = meta[k]
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# ============================================================
# Gráficos Pullback + Trades
# ============================================================

def plot_pullbacks_on_price(
    dfs: Dict[str, pd.DataFrame],
    pullbacks_df: pd.DataFrame,
    symbol: str,
    style: GreenStyle,
    output_dir: str,
    trades_df: pd.DataFrame | None = None,
    max_charts: int = 50,
):
    if pullbacks_df is None or pullbacks_df.empty:
        print(f"[{symbol}] No hay pullbacks para graficar.")
        return

    tf_pb = getattr(style, "pullback_tf", "15m")
    df_price = _choose_plot_df(dfs, tf_pb)
    if df_price is None or df_price.empty:
        print(f"[{symbol}] No encontré datos de precio para graficar pullbacks.")
        return

    os.makedirs(output_dir, exist_ok=True)
    sym_clean = symbol.replace("/", "")

    df_price = df_price.copy()
    df_price["timestamp"] = pd.to_datetime(df_price["timestamp"])

    pb_df = pullbacks_df.copy().reset_index(drop=True)
    if len(pb_df) > max_charts:
        print(f"[{symbol}] Hay {len(pb_df)} pullbacks, solo graficaré los primeros {max_charts}.")
        pb_df = pb_df.head(max_charts)

    if trades_df is not None and not trades_df.empty:
        trades_df = trades_df.copy()
        trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"])
        trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"])
        if "impulse_start" in trades_df.columns:
            trades_df["impulse_start"] = pd.to_datetime(trades_df["impulse_start"])

    for idx, row in pb_df.iterrows():
        try:
            pb_start = pd.to_datetime(row.get("pb_start"))
            pb_end = pd.to_datetime(row.get("pb_end"))
            imp_start = pd.to_datetime(row.get("impulse_start"))
            imp_end = pd.to_datetime(row.get("impulse_end"))
        except Exception:
            continue

        if pd.isna(pb_start) or pd.isna(pb_end) or pd.isna(imp_start) or pd.isna(imp_end):
            continue

        sr_level = float(row.get("sr_level", row.get("impulse_sr_level", 0.0)))
        direction = row.get("direction", row.get("impulse_direction", ""))

        win_start = min(imp_start, pb_start) - timedelta(hours=24)
        win_end = max(imp_end, pb_end) + timedelta(hours=24)

        df_win = df_price[
            (df_price["timestamp"] >= win_start) &
            (df_price["timestamp"] <= win_end)
        ].copy()
        if df_win.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_win["timestamp"], df_win["close"], label=f"Precio {tf_pb}")

        # SR + tolerancia
        if sr_level > 0:
            ax.axhline(sr_level, linestyle="--", linewidth=1.0,
                       label=f"SR {sr_level:.2f}")
            tol = style.pullback_sr_tolerance
            sr_upper = sr_level * (1 + tol)
            sr_lower = sr_level * (1 - tol)
            ax.axhline(sr_upper, color="orange", lw=1.0, linestyle=":",
                       alpha=0.7, label=f"SR + tol ({tol*100:.1f}%)")
            ax.axhline(sr_lower, color="orange", lw=1.0, linestyle=":",
                       alpha=0.7, label=f"SR - tol ({tol*100:.1f}%)")

        # Impulse start/end
        ax.axvline(imp_start, color="black", linestyle=":", linewidth=1.0, label="Impulse start")
        ax.axvline(imp_end, color="black", linestyle="--", linewidth=1.0, label="Impulse end")

        # Pullback start/end
        ax.axvline(pb_start, color="green", linestyle=":", linewidth=1.2, label="Pullback start")
        ax.axvline(pb_end, color="red", linestyle=":", linewidth=1.2, label="Pullback end")

        # sr_touch_ts
        sr_touch_ts = row.get("sr_touch_ts")
        sr_touch_price = row.get("sr_touch_price", sr_level if sr_level > 0 else None)
        if pd.notna(sr_touch_ts) and sr_touch_price is not None:
            st_ts = pd.to_datetime(sr_touch_ts)
            ax.scatter([st_ts], [float(sr_touch_price)], marker="o", s=40, label="SR touch")

        # rebound_ts
        rebound_ts = row.get("rebound_ts")
        if pd.notna(rebound_ts):
            rb_ts = pd.to_datetime(rebound_ts)
            row_rb = df_price.iloc[(df_price["timestamp"] - rb_ts).abs().argsort()[:1]]
            if not row_rb.empty:
                ax.scatter(
                    [rb_ts],
                    [float(row_rb["close"].iloc[0])],
                    marker="^",
                    s=60,
                    label="Rebound candle",
                )

        # Trades que corresponden a ESTE impulso + dirección y ocurren luego del fin del pullback
        if trades_df is not None and not trades_df.empty and "impulse_start" in trades_df.columns:
            trades_pb = trades_df[
                (trades_df["impulse_start"] == imp_start) &
                (trades_df["direction"] == direction) &
                (trades_df["entry_time"] >= pb_end)
            ].copy()
            if not trades_pb.empty:
                draw_trades_on_ax(ax, trades_pb)

        ax.set_title(
            f"{symbol} {style.name} | Pullback #{idx+1} "
            f"| dir={direction} | TF={tf_pb}"
        )
        ax.set_xlabel("Tiempo")
        ax.set_ylabel("Precio")
        ax.legend(loc="best")
        ax.grid(True)

        out_path = os.path.join(
            output_dir,
            f"pullback_{sym_clean}_{style.name}_#{idx+1}.png",
        )
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()


# ============================================================
# Backtest runner – SIEMPRE flujo completo
# ============================================================

def run_backtest(
    symbol: str,
    style: GreenStyle,
    cfg: Any,
    input_dir: str,
    output_dir: str,
    days: int = 360,
    ai_supervisor: bool = False,
    saveResults: bool = True,
):
    print(f"\n========== BACKTEST {symbol} ({style.name}) ==========")

    dfs = load_timeframes_for_symbol(symbol, base_dir=input_dir)
    if not dfs or all((df is None or df.empty) for df in dfs.values()):
        print(f"[{symbol}] ERROR: No se pudieron cargar velas desde {input_dir}.")
        return None

    print(f"[{symbol}] Recortando últimos {days} días...")
    dfs = filter_last_days(dfs, days=days)

    exchange = BacktestExchange(candles_by_symbol={symbol: dfs})
    impulse_strategy = DefaultImpulseStrategy()
    pullback_strategy = DefaultPullbackStrategy()
    trigger_ai_engine = DefaultTriggerAIStrategy(use_ai=True)
    position_strategy = DefaultPositionStrategy(exchange=exchange)

    stages = GreenStages(
        impulse=impulse_strategy,
        pullback=pullback_strategy,
        trigger_ai=trigger_ai_engine,
        position=position_strategy,
    )

    core = GreenV3Core(
        style=style,
        stages=stages,
    )

    os.makedirs(output_dir, exist_ok=True)
    sym_clean = symbol.replace("/", "")

    print(f"[{symbol}] Ejecutando GreenV3Core.run() (flujo completo)...")
    trades = core.run(
        symbol=symbol,
        dfs=dfs,
        cfg=cfg,
    )

    if not trades:
        print(f"[{symbol}] No se generaron trades.")
        return None

    print(f"[{symbol}] Backtest finalizado, trades={len(trades)}")

    df_trades = pd.DataFrame([asdict(t) for t in trades])

    if saveResults:
        # CSV trades + equity / histograma
        out_csv = os.path.join(output_dir, f"backtest_{sym_clean}_{style.name}.csv")
        df_trades.to_csv(out_csv, index=False)
        print(f"[{symbol}] Trades guardados en {out_csv} ({len(df_trades)} filas).")
        plot_equity_and_hist(df_trades, symbol, style, output_dir)

        # Reconstruir pullbacks SOLO para gráficos (no afecta core)
        df_pb = build_pullbacks_dataframe(
            symbol=symbol,
            dfs=dfs,
            style=style,
            impulse_strategy=impulse_strategy,
            pullback_strategy=pullback_strategy,
        )
        if not df_pb.empty:
            out_pb_csv = os.path.join(output_dir, f"pullbacks_{sym_clean}_{style.name}.csv")
            df_pb.to_csv(out_pb_csv, index=False)
            print(f"[{symbol}] Pullbacks guardados en {out_pb_csv} ({len(df_pb)} filas).")

            plot_pullbacks_on_price(
                dfs=dfs,
                pullbacks_df=df_pb,
                symbol=symbol,
                style=style,
                output_dir=output_dir,
                trades_df=df_trades,
            )
        else:
            print(f"[{symbol}] No se detectaron pullbacks para graficar.")

    return df_trades


# ============================================================
# OPTIMIZER
# ============================================================

def run_optimizer(
    symbol: str,
    base_style: GreenStyle,
    input_dir: str,
    output_dir: str,
    days: int = 360,
    ai_supervisor: bool = False,
):
    if base_style.name.upper() != "DAY":
        print(f"[{symbol}] WARNING: el optimizador está pensado para DAY, estilo actual={base_style.name}")
    
    param_grid: Dict[str, List[Any]] = {
        "impulse_min_body_pct":   [0.04, 0.05, 0.06],
        "impulse_break_pct":      [0.005, 0.01, 0.015],
        "pullback_sr_tolerance":  [0.003, 0.005, 0.007],
        "pullback_slow_body_factor": [0.4, 0.5, 0.6],
        "trigger_sr_tolerance":   [0.003, 0.005, 0.007],
        "trigger_entry_ema_length": [8, 20, 50],
        "trigger_entry_body_min_ratio": [0.4, 0.5, 0.6],
        "entry_sl_buffer_pct":    [0.0005, 0.001],
        "entry_min_rr":           [1.5, 2.0, 2.5],
        "position_ema_trail_span": [8, 20, 50],
    }

    results_rows: List[Dict[str, Any]] = []

    for params in iter_param_grid(param_grid):
        print("\n===============================================")
        print(f"[{symbol}] Probando config: {params}")
        print("===============================================")

        style_cfg = replace(base_style, **params)

        trades_df = run_backtest(
            symbol=symbol,
            style=style_cfg,
            cfg=style_cfg,
            input_dir=input_dir,
            output_dir=output_dir,
            days=days,
            ai_supervisor=ai_supervisor,
            saveResults=False,
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
    parser.add_argument("--optimize", action="store_true", help="Ejecutar grid search de parámetros (optimizer)")
    args = parser.parse_args()

    symbol = args.symbol
    style = get_style(args.style)
    cfg = style
    use_ai = args.ai  # hoy no lo usamos en trigger_ai, queda para futuro

    if args.optimize:
        run_optimizer(
            symbol=symbol,
            base_style=style,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            days=args.days,
            ai_supervisor=use_ai,
        )
    else:
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