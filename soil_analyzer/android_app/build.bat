@echo off
echo ==========================================
echo Soil Health Analyzer - Android Build Tool
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    goto :EOF
)

REM Check if buildozer is installed
pip show buildozer >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing buildozer...
    pip install buildozer
)

REM Create virtual environment if not exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo.
echo Choose an option:
echo 1. Run the app locally for testing
echo 2. Build Android APK (debug mode)
echo 3. Build Android APK (release mode)
echo 4. Clean build files
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo Running app locally...
    python main.py
) else if "%choice%"=="2" (
    echo Building Android debug APK...
    buildozer android debug
    echo APK should be available in the bin directory
) else if "%choice%"=="3" (
    echo Building Android release APK...
    buildozer android release
    echo APK should be available in the bin directory
) else if "%choice%"=="4" (
    echo Cleaning build files...
    buildozer android clean
    echo Clean completed
) else if "%choice%"=="5" (
    echo Exiting...
    goto :EOF
) else (
    echo Invalid choice!
)

echo.
echo Done!
pause 