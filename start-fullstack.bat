@echo off
echo Starting Talksmith - Full-Stack Architecture
echo ===============================================

echo.
echo Starting AI Service (FastAPI)...
start "AI Service" cmd /k "cd ai-service && python -m uvicorn main:app --reload --port 8000"

timeout /t 3 /nobreak >nul

echo Starting Next.js Full-Stack App (Frontend + API)...
start "Next.js App" cmd /k "npm run dev"

echo.
echo Services are starting...
echo - AI Service: http://localhost:8000
echo - Next.js App: http://localhost:3000
echo.
echo Press any key to exit...
pause >nul
