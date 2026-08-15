@echo off
REM ============================================================
REM  STEP 1 of 2 - Google Cloud login
REM  Just double-click this file. A browser window will open.
REM ============================================================
setlocal

set "GCLOUD=%USERPROFILE%\gcloud-sdk\google-cloud-sdk\bin\gcloud.cmd"
set "PROJECT=project-387a4739-379e-4c55-a6f"

if not exist "%GCLOUD%" (
  echo.
  echo   ERROR: gcloud not found at:
  echo   %GCLOUD%
  echo.
  pause
  exit /b 1
)

echo.
echo  ============================================================
echo   A browser will open. Sign in with the SAME Google account
echo   you used to create the project.
echo.
echo   Your console link had "authuser=1", which means it was your
echo   SECOND Google account -- pick that one, not the first.
echo  ============================================================
echo.
pause

call "%GCLOUD%" auth login

echo.
echo  Setting project to %PROJECT% ...
call "%GCLOUD%" config set project %PROJECT%

echo.
echo  ============================================================
echo   RESULT - copy the lines below and send them to Claude:
echo  ============================================================
call "%GCLOUD%" config get-value project
call "%GCLOUD%" auth list
echo  ============================================================
echo.
pause
