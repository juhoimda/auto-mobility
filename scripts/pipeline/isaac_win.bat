@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM  Isaac Sim Digital Twin Mesh Verification Launcher (Windows Host)
REM ==========================================================

if "%~1"=="" (
    echo ==========================================================
    echo  사용법: %~nx0 MESH_FILE_NAME [--headless] [--no-physics] [--scale SCALE]
    echo  예시  : %~nx0 session_20260804_mesh.obj
    echo  예시  : %~nx0 ros2_data\meshes\session_20260804_mesh.obj --scale 1.0
    echo ==========================================================
    exit /b 1
)

set MESH_INPUT=%~1
shift

REM 프로젝트 루트 및 Mesh 경로 계산
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..\..
set MESH_DIR=%PROJECT_DIR%\ros2_data\meshes
set LOADER_SCRIPT=%PROJECT_DIR%\src\auto_mobility\processing\load_isaac_mesh.py

if exist "%MESH_INPUT%" (
    set MESH_FILE=%MESH_INPUT%
) else if exist "%MESH_DIR%\%MESH_INPUT%" (
    set MESH_FILE=%MESH_DIR%\%MESH_INPUT%
) else if exist "%PROJECT_DIR%\%MESH_INPUT%" (
    set MESH_FILE=%PROJECT_DIR%\%MESH_INPUT%
) else (
    set MESH_FILE=%MESH_INPUT%
)

if not exist "%MESH_FILE%" (
    echo ❌ 오류: Mesh 파일을 찾을 수 없습니다 -^> %MESH_FILE%
    echo 💡 팁: %MESH_DIR% 디렉터리에 Mesh 파일이 있는지 확인하세요.
    exit /b 1
)

echo ==========================================================
echo  🚀 Digital Twin Isaac Sim Launcher (Windows Host)
echo  📁 Input Mesh : %MESH_FILE%
echo  📜 Python App : %LOADER_SCRIPT%
echo ==========================================================

REM Windows Omniverse / Isaac Sim Python 실행기 탐색
set ISAAC_PYTHON=

for /d %%D in ("%LOCALAPPDATA%\ov\pkg\isaac-sim-*", "%LOCALAPPDATA%\ov\pkg\isaac_sim-*", "C:\isaac-sim-*") do (
    if exist "%%D\python.bat" (
        set ISAAC_PYTHON=%%D\python.bat
    )
)

if defined ISAAC_PYTHON (
    echo 🔍 Isaac Sim Windows Python Found: %ISAAC_PYTHON%
    "%ISAAC_PYTHON%" "%LOADER_SCRIPT%" "%MESH_FILE%" %1 %2 %3 %4 %5 %6 %7 %8 %9
) else (
    echo ⚠️ Omniverse Isaac Sim 설치 경로를 찾지 못했습니다. 기본 python으로 시도합니다.
    python "%LOADER_SCRIPT%" "%MESH_FILE%" %1 %2 %3 %4 %5 %6 %7 %8 %9
)
