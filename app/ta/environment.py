import os, sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ENV_PATH = os.path.join(BASE_DIR, ".env")

def load_env_file(path: str):
    if not os.path.exists(path):
        print(f"[ENV] Archivo .env no encontrado en: {path}")
        return
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.split("#", 1)[0]
                val = val.strip().replace('"', "").replace("'", "")
                os.environ[key] = val
        print(f"[ENV] Variables cargadas desde {path}")
    except Exception as e:
        print(f"[ENV] Error cargando .env: {e}")


load_env_file(ENV_PATH)
sys.path.append(BASE_DIR)
