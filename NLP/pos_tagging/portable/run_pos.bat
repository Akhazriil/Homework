@echo off
cd /d "%~dp0"
chcp 65001 >nul
set /p "TEXT=Введите слово или предложение: "
pos_tagger.exe --text "%TEXT%"
echo.
pause
