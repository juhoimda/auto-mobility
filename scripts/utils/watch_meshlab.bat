@echo off
title WSL MeshLab Auto-Launcher
echo ========================================================
echo   WSL -> Windows MeshLab Auto-Launcher Running...
echo   WSL에서 view_mesh.sh 실행 시 자동으로 MeshLab을 띄웁니다.
echo ========================================================

set "MESHLAB=C:\Program Files\VCG\MeshLab\meshlab.exe"
set "REQUEST_FILE=C:\ubuntu_shared\view_request.txt"

if not exist "%MESHLAB%" (
    echo [경고] %MESHLAB% 경로를 찾을 수 없습니다.
    echo MeshLab 설치 위치를 확인해주세요.
)

:loop
if exist "%REQUEST_FILE%" (
    set /p TARGET_FILE=<"%REQUEST_FILE%"
    del "%REQUEST_FILE%" >nul 2>&1
    if defined TARGET_FILE (
        echo [실행] MeshLab으로 여는 중: %TARGET_FILE%
        start "" "%MESHLAB%" "%TARGET_FILE%"
    )
)
timeout /t 1 /nobreak >nul
goto loop
