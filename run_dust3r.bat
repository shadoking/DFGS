@echo off
setlocal enabledelayedexpansion
set base_path=data

for /d %%F in ("%base_path%\*") do (
    set "folder_path=%%F"
    for %%I in ("%%F") do set "folder_name=%%~nI"

    if "!folder_name!" neq "!folder_name:_2=!" (
        echo Process train: !folder_name!
        python script_for_dust3r.py -s !folder_path!
    )
)
echo ALL DONE!
