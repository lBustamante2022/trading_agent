# app/utils/sound.py
import platform
import subprocess
import sys

def play_beep():
    """
    Beep confiable en Mac.
    - macOS: usa 'afplay' con un sonido del sistema.
    - Linux/otros: campanita en consola.
    - Windows: campanita simple (si querés, podés mejorarla).
    """
    system = platform.system()

    try:
        if system == "Darwin":  # macOS
            # Sonidos típicos: Ping.aiff, Glass.aiff, Submarine.aiff, etc.
            subprocess.run(
                ["afplay", "/System/Library/Sounds/Glass.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        elif system == "Windows":
            import winsound
            winsound.MessageBeep()
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        # fallback silencioso si algo falla
        sys.stdout.write("\a")
        sys.stdout.flush()