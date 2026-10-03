@echo off
rem Creates a "VisionTrack AI" shortcut on your Desktop that starts the app and opens the dashboard.
cd /d "%~dp0"
powershell -NoProfile -Command ^
  "$d=[Environment]::GetFolderPath('Desktop');" ^
  "$s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'VisionTrack AI.lnk'));" ^
  "$s.TargetPath='%~dp0run.bat'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%SystemRoot%\System32\shell32.dll,22';" ^
  "$s.Description='Start VisionTrack AI'; $s.Save()"
echo Shortcut "VisionTrack AI" created on your Desktop.
echo You can delete the old "VisionTrack AI.url" shortcut - the new one starts the server by itself.
pause
