@REM @echo off
@REM setlocal enabledelayedexpansion
@REM set base_path=data

@REM for /d %%F in ("%base_path%\*") do (
@REM     set "folder_path=%%F"
@REM     for %%I in ("%%F") do set "folder_name=%%~nI"

@REM     if "!folder_name!" neq "!folder_name:_8=!" (
@REM         set "test_folder_name=!folder_name:_8=_rest!"
@REM         echo Process depth: !test_folder_name!
@REM         python Depth-Anything-V2/run.py --encoder vitl --pred-only --grayscale --img-path %base_path%\!test_folder_name!\images --outdir %base_path%\!test_folder_name!\depths
@REM         python utils/make_depth_scale.py --base_dir %base_path%\!test_folder_name! --depths_dir %base_path%\!test_folder_name!\depths
@REM     )
@REM )
@REM echo ALL DONE!


@echo off
setlocal enabledelayedexpansion
set base_path=data

for /d %%F in ("%base_path%\*") do (
    set "folder_path=%%F"
    for %%I in ("%%F") do set "folder_name=%%~nI"

    @REM echo Process train: !folder_name!

    @REM python train.py -s !folder_path! -m output0/!folder_name! -d depths
    @REM python train.py -s !folder_path! -m output1/!folder_name! -d depths --lambda_diffusion 0

    if "!folder_name!" neq "!folder_name:_8=!" (
        set "test_folder_name=!folder_name:_8=_rest!"
        echo Process render: !folder_name!
        python render.py -s %base_path%\!test_folder_name! -m output0/!folder_name!
        python render.py -s %base_path%\!test_folder_name! -m output1/!folder_name!

        echo Process metrics: !folder_name!
        python metrics.py -m output0/!folder_name!
        python metrics.py -m output1/!folder_name!
    )
)
echo ALL DONE!
