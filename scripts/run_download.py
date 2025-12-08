# scripts/binance_download.py
"""
Descarga datos OHLCV de Binance para múltiples símbolos y timeframes
y los guarda en CSV dentro de la carpeta ./data

Formato de archivo:
    data/binance_<SYMBOL_SIN_SLASH>_<TIMEFRAME>.csv
Ejemplo:
    data/binance_BTCUSDT_1h.csv
"""

import os
import time
from datetime import datetime

import ccxt
import pandas as pd

# -------- CONFIGURACIÓN --------

SYMBOLS = [
    "ADA/USDT",
    "AVAX/USDT",
    "BCH/USDT",
    "BNB/USDT",
    "BTC/USDT",
    "ETH/USDT",
    "HBAR/USDT",
    "LINK/USDT",
    "LTC/USDT",
    "SHIB/USDT",
    "SOL/USDT",
    "TRX/USDT",
    "XRP/USDT",
]

TIMEFRAMES = [
    "1w",
    "1d",
    "4h",
    "1h",   
    "15m", 
    "30m", 
    "5m",  
    "3m",
    "1m",  
]

# Fecha inicial de descarga (podés ajustar a gusto)
SINCE_STR = "2023-01-01 00:00:00"
SINCE_MS = int(datetime.strptime(SINCE_STR, "%Y-%m-%d %H:%M:%S").timestamp() * 1000)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "input")
os.makedirs(DATA_DIR, exist_ok=True)

# Límite máximo por llamada en Binance (ccxt)
LIMIT = 1000

exchange = ccxt.binance({
    "enableRateLimit": True,
})


def symbol_to_filename(symbol: str, timeframe: str) -> str:
    sym_clean = symbol.replace("/", "")
    fname = f"binance_{sym_clean}_{timeframe}.csv"
    return os.path.join(DATA_DIR, fname)


def download_ohlcv(symbol: str, timeframe: str):
    """
    Descarga incremental:
    - Si no existe el CSV: baja todo desde SINCE_MS.
    - Si existe: continúa desde el último timestamp + 1.
    """

    path = symbol_to_filename(symbol, timeframe)
    print(f"\n=== {symbol} {timeframe} ===")
    print(f"Archivo destino: {path}")

    since = SINCE_MS
    all_data = []

    if os.path.exists(path):
        # Reanudar desde el último registro + 1ms
        df_existing = pd.read_csv(path)
        if "timestamp" in df_existing.columns:
            last_ts = int(df_existing["timestamp"].max())
            print(f"CSV existente, último ts = {last_ts}")
            since = last_ts + 1
            all_data = df_existing.values.tolist()
        else:
            print("CSV sin columna 'timestamp', re-descargando todo.")

    while True:
        print(f"Solicitando {symbol} {timeframe} desde {since}...")
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=LIMIT)
        if not ohlcv:
            print("No hay más datos nuevos.")
            break

        all_data.extend(ohlcv)
        last_ts = ohlcv[-1][0]
        print(f"Recibidos {len(ohlcv)} registros. Último ts = {last_ts}")
        since = last_ts + 1

        # Pausa para respetar rate limit
        time.sleep(exchange.rateLimit / 1000)

    if not all_data:
        print("No se descargó nada nuevo.")
        return

    df = pd.DataFrame(
        all_data,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    # En ms -> convertir a datetime legible, pero mantener timestamp numérico también
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    df.to_csv(path, index=False)
    print(f"Guardado CSV con {len(df)} filas en {path}")


def main():
    print("=== Descarga histórica Binance ===")
    print(f"Desde: {SINCE_STR} (ms={SINCE_MS})")
    print(f"Carpeta de datos: {DATA_DIR}")

    for sym in SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                download_ohlcv(sym, tf)
            except Exception as e:
                print(f"❌ Error descargando {sym} {tf}: {e}")
                # Pequeña pausa extra en caso de error
                time.sleep(5)


if __name__ == "__main__":
    main()