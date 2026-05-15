@echo off
title Car-Manual-RAG Launcher
color 0B

echo ==========================================
echo    Starting Car-Manual-RAG System
echo ==========================================

:: 1. Start Backend
echo [*] Launching FastAPI backend service in the background...
start /B python backend/main.py

:: Give models time to load
echo [*] Waiting for models to load...
timeout /t 5 /nobreak > nul

:: 2. Start Frontend
echo [*] Launching Streamlit frontend interface...
streamlit run frontend/app.py

echo.
echo [!] Note: To stop the system, close this window and manually terminate any remaining python processes if necessary.
pause