@echo off
echo ========================================
echo   Bank Fraud Detection - Build Script
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Build the executable
echo.
echo Building executable...
pyinstaller build.spec --clean

echo.
echo ========================================
echo   Build Complete!
echo   Output: dist\FraudDetection.exe
echo ========================================
pause
