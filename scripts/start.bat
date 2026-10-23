@echo off
chcp 65001 >nul
rem 자비서 시작 (윈도우). 더블클릭하거나 바탕화면 바로가기로 연다.  담당: 문태현 (배포)
rem 하는 일은 scripts\start.py 에 있다. 여기서는 conda 환경 JARVISEO 의 파이썬만 찾는다.
rem 다른 곳에 깔았으면 시스템 환경변수 JARVISEO_PYTHON 에 python.exe 경로를 적는다.

cd /d "%~dp0.."

set "PY=%JARVISEO_PYTHON%"
for %%D in ("%USERPROFILE%\.conda" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3") do (
    if not defined PY if exist "%%~D\envs\JARVISEO\python.exe" set "PY=%%~D\envs\JARVISEO\python.exe"
)
if not defined PY (
    echo conda 환경 JARVISEO 를 찾지 못했습니다. docs\deploy.md 의 설치 순서를 따르세요.
    pause
    exit /b 1
)

rem conda activate 를 하지 않고 켜서, 환경의 DLL 폴더를 PATH 앞에 직접 넣는다.
for %%I in ("%PY%") do set "ENVDIR=%%~dpI"
set "PATH=%ENVDIR%;%ENVDIR%Library\bin;%ENVDIR%Scripts;%PATH%"

"%PY%" scripts\start.py %*
echo.
pause
