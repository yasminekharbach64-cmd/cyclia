"""
Gunicorn configuration for CyclIA production deployment.

Servidor WSGI de producción para Flask. Se usa en Docker/Render
en lugar del servidor de desarrollo de Flask (que no es apto para producción).
"""

import os

# === Server socket ===
# Puerto dinámico (Render asigna automáticamente vía variable $PORT)
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# === Worker processes ===
# 2 workers para tier gratuito de Render (512MB RAM).
# En producción con más recursos: (2 * CPU_count) + 1
workers = 1

# Tipo de worker: sync (síncrono) — adecuado para cargas moderadas
# y para modelos que mantienen estado en memoria (embeddings PubMed).
worker_class = "sync"

# Threads por worker (mejora concurrencia sin gastar más RAM)
threads = 2

# === Timeouts ===
# Tiempo máximo por petición. Generoso porque la primera query
# carga embeddings (~10-15s) y luego se mantienen en memoria.
timeout = 120

# Mantener conexiones abiertas para reutilización
keepalive = 5

# === Logging ===
accesslog = "-"   # stdout (Docker/Render capturan logs desde ahí)
errorlog = "-"    # stderr
loglevel = "info"

# === Preload ===
# Carga la app antes de hacer fork de workers.
# Ahorra RAM porque los embeddings se cargan UNA vez y se comparten.
preload_app = True