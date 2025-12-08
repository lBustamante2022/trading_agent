# app/utils/discord_notifier.py
import os
import requests

def send_discord_alert(
    title: str,
    description: str,
    color: int = 0x00FF00,
    webhook_env_var: str = "DISCORD_WEBHOOK_URL",
):
    """
    Envía una alerta a Discord usando un Webhook.
    El Webhook se toma de la variable de entorno DISCORD_WEBHOOK_URL.
    """
    webhook_url = os.getenv(webhook_env_var)
    if not webhook_url:
        print("⚠️  DISCORD_WEBHOOK_URL no está configurado, no se envía alerta.")
        return

    payload = {
        "embeds": [
            {
                "title": title,
                "description": description,
                "color": color,
            }
        ]
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=5)
        if resp.status_code >= 300:
            print(f"⚠️  Error al enviar alerta Discord: {resp.status_code} {resp.text}")
    except Exception as e:
        print("⚠️  Excepción enviando alerta Discord:", e)