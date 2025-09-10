@echo off
echo Installing Talksmith Full-Stack Dependencies
echo =============================================

echo.
echo Installing Frontend Dependencies (Next.js + API)...
call npm install
if %errorlevel% neq 0 (
    echo Error installing frontend dependencies!
    pause
    exit /b 1
)

echo.
echo Installing AI Service Dependencies (Python)...
cd ai-service
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing AI service dependencies!
    pause
    exit /b 1
)

cd ..
echo.
echo ✅ All dependencies installed successfully!
echo.
echo Next steps:
echo 1. Make sure your OpenAI API key is set in .env
echo 2. Run start-fullstack.bat to start both services
echo 3. Open http://localhost:3000 in your browser
pause
