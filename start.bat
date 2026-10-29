@echo off
title PDF QA Assistant - Topic #9 (Ollama + TF-IDF)
cd /d "%~dp0"

echo ================================================================
echo    PDF Question Answering Assistant (Topic #9)
echo    Powered by TF-IDF + Cosine Similarity and Local Ollama
echo ================================================================
echo.

:: 1. Check if Ollama is active on port 11434
echo [*] Checking Ollama local server status...
netstat -ano | findstr 11434 >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] Ollama is not running. Starting Ollama in the background...
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        start "" /min "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve
    ) else (
        start "" /min "ollama" serve
    )
    timeout /t 4 /nobreak >nul
    echo [OK] Ollama started successfully.
) else (
    echo [OK] Ollama is already running on port 11434.
)

:: 2. Activate Python Virtual Environment
echo.
echo [*] Checking Python environment...
if exist "venv\Scripts\activate.bat" (
    echo [OK] Activating virtual environment (venv)...
    call "venv\Scripts\activate.bat"
) else (
    echo [!] Virtual environment not found. Setting up venv...
    python -m venv venv
    call "venv\Scripts\activate.bat"
    echo [*] Installing dependencies from requirements.txt...
    pip install -r requirements.txt
)

:: 3. Launch the Streamlit application
echo.
echo ================================================================
echo    Opening PDF QA Assistant at http://localhost:8501
echo    Press Ctrl+C in this window to stop the server
echo ================================================================
echo.
python -m streamlit run app.py

pause
