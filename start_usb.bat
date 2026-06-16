@echo off
color 0a
title Pornire Platforma Voice Conversion (USB)

echo ===================================================
echo   Verificare si pornire aplicatie Voice Conversion
echo ===================================================
echo.

:: Verifica daca Python este instalat pe noul PC
python --version >nul 2>&1
if %errorlevel% neq 0 (
    color 0c
    echo [EROARE] Python nu este instalat pe acest calculator!
    echo Te rugam sa instalezi Python (minim versiunea 3.10) si sa il adaugi in PATH.
    pause
    exit /b
)

:: Numele mediului virtual (pe PC-ul nou)
set VENV_NAME=venv_app

:: Verifica daca mediul virtual exista deja pe noul PC
if not exist "%VENV_NAME%\Scripts\activate.bat" (
    echo [INFO] Se creeaza un nou mediu virtual local pentru acest calculator...
    python -m venv %VENV_NAME%
    if %errorlevel% neq 0 (
        color 0c
        echo [EROARE] Nu s-a putut crea mediul virtual.
        pause
        exit /b
    )
    echo [OK] Mediu virtual creat cu succes.
    
    :: Instaleaza pachetele deoarece este prima rulare pe acest PC
    echo.
    echo [INFO] Se instaleaza dependentele necesare... Poate dura cateva minute.
    call "%VENV_NAME%\Scripts\activate.bat"
    python -m pip install --upgrade pip
    pip install -r requirements_final.txt
    if %errorlevel% neq 0 (
        color 0e
        echo [AVERTISMENT] Unele pachete ar fi putut da eroare la instalare, dar incercam sa continuam.
    )
) else (
    echo [OK] Mediul virtual "%VENV_NAME%" gasit.
)

:: Porneste aplicatia
echo.
echo [INFO] Se porneste serverul web...
call "%VENV_NAME%\Scripts\activate.bat"

:: Comanda ta de rulare
python -m uvicorn webapp.backend.app:app --host 0.0.0.0 --port 8000

echo.
pause
