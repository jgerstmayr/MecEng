@echo off
REM Deletes all .npz files found under any fMBSLLM subdirectory of each LLM run log_* folder
REM adapt the path to the log_* folder accordingly (e.g. results\logsFlexuAi_API_v1)

setlocal
set "ROOT=%~dp0..\..\results\logsFlexuAi_API_v1\"

for /d %%D in ("%ROOT%log_*") do (
    if exist "%%D\fMBSLLM" (
        echo Deleting .npz files in "%%D\fMBSLLM" ...
        del /s /q "%%D\fMBSLLM\*.npz"
    )
)

echo Done.
endlocal
