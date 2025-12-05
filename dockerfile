# Imagen base ligera
FROM python:3.10-slim

# Evitar buffers en logs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=America/Argentina/Buenos_Aires

# Actualizamos sistema y herramientas básicas
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Directorio de trabajo
WORKDIR /app

# Instalamos dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del proyecto
COPY . .

# (Opcional) si usás instalación editable
# RUN pip install --no-cache-dir -e .

# Variables de entorno para OKX (las completás al correr el contenedor)
# ENV OKX_API_KEY=tu_key
# ENV OKX_API_SECRET=tu_secret
# ENV OKX_API_PASSPHRASE=tu_pass
# ENV OKX_ENV=demo   # o 'live'

# === Comando por defecto ===
# Para correr el trader DAY en modo demo:
CMD ["python", "-m", "scripts.green_day_live_okx_trader"]

# Si quisieras exponer la API FastAPI en vez del trader,
# comentar la línea anterior y descomentar esta:
# CMD ["python", "main.py"]