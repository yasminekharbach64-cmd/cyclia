python api.pystart_ollama.bat@echo off
start "Ollama Server" cmd /k "ollama serve"
timeout /t 3 /nobreak >nul
start "Ollama Chat" cmd /k "ollama run mistral"