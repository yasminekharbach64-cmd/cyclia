"""
Gunicorn configuration for CyclIA production deployment.
Servidor WSGI de produccion para Flask. Se usa en Docker/Render
en lugar del servidor de desarrollo de Flask (que no es apto para produccion).
"""
import os

# === Server socket ===
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# === Worker processes ===
# 1 worker para tier gratuito de Render (512MB RAM).
workers = 1
worker_class = "sync"
threads = 2

# === Timeouts ===
timeout = 120
keepalive = 5

# === Logging ===
accesslog = "-"
errorlog = "-"
loglevel = "info"

# === Preload ===
preload_app = True
