@echo off
setlocal enabledelayedexpansion
title Advocium - LexiQA (Legal QA & Compliance Intelligence - Topic #9)
cd /d "%~dp0"

echo ================================================================
echo    Advocium · LexiQA (Legal QA Engine - Topic #9)
echo    TF-IDF Vector Space + Cosine Similarity Information Retrieval
echo    Team: Skill G
echo ================================================================
echo.

:: 1. Launch Python FastAPI Serverless Backend in the background
echo [*] Starting Python NLP Backend on http://127.0.0.1:8000...
start "LexiQA Python Backend" /min cmd /c "python -m uvicorn api.index:app --port 8000"

:: 2. Launch Vite React Frontend
echo [*] Starting React Frontend on http://localhost:3000...
start "" http://localhost:3000
npm run dev

pause
