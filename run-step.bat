@echo off
rem Один шаг пейпер-трейдинга. Запускается планировщиком каждые 15 минут.
rem Если Python не в PATH — задайте переменную окружения PYTHON полным путём.

setlocal
cd /d "%~dp0"

if "%PYTHON%"=="" set PYTHON=python

echo. >> paper\step.log
echo [%DATE% %TIME%] step >> paper\step.log
"%PYTHON%" -m lab.paper step >> paper\step.log 2>&1
