@echo off
setlocal enabledelayedexpansion
title PDF QA Assistant (Topic #9)
cd /d "%~dp0"

echo ================================================================
echo    PDF Question Answering Assistant (Topic #9)
echo    TF-IDF + Cosine Similarity ^| Local Ollama
echo ================================================================
echo.

:: 1. Verify Virtual Environment
if not exist "%~dp0venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at:
    echo         "%~dp0venv"
    echo.
    echo Please create it first by running:
    echo   python -m venv venv
    echo   venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: 2. Check and Start Ollama Server
echo [*] Checking Ollama server (http://127.0.0.1:11434)...
netstat -ano | findstr 11434 >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Starting Ollama in the background...
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        start "" /b "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    ) else (
        start "" /b ollama serve
    )
    :: Safe 3-second delay using ping
    ping 127.0.0.1 -n 4 >nul
    echo [OK] Ollama service started.
) else (
    echo [OK] Ollama is already running on port 11434.
)

:: 3. Launch Streamlit Application
echo.
echo ================================================================
echo    Starting Streamlit Application...
echo    Opening http://localhost:8501 in your default browser
echo    (Press Ctrl+C in this window to stop the server)
echo ================================================================
echo.

"%~dp0venv\Scripts\python.exe" -m streamlit run "%~dp0app.py" --browser.gatherUsageStats=false

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Streamlit exited with an error code: %errorlevel%
)

echo.
pause
