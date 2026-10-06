@echo off
echo ============================================
echo Сборка ABR Converter в EXE
echo ============================================
echo.

if not exist "venv" (
    echo Создание виртуального окружения...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Ошибка создания виртуального окружения
        pause
        exit /b 1
    )
)

echo Установка зависимостей...
call venv\Scripts\activate.bat
pip install openpyxl pyinstaller
if %errorlevel% neq 0 (
    echo Ошибка установки зависимостей
    pause
    exit /b 1
)

echo.
echo Запуск сборки...
pyinstaller --onefile --windowed --name "ABR_Converter" --add-data "abris_xl.py;." --hidden-import openpyxl --hidden-import openpyxl.cell --hidden-import openpyxl.styles abris_converter_gui.py

if %errorlevel% neq 0 (
    echo.
    echo Ошибка сборки!
    pause
    exit /b 1
)

echo.
echo ============================================
echo Сборка завершена!
echo EXE файл находится в папке: dist\ABR_Converter.exe
echo ============================================
pause
