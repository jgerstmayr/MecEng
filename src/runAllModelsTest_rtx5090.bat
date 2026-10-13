@echo off
REM Runs the LLM model creation tests reported in the paper (Windows, tested with an RTX 5090, 32 GB).
REM Usage (from the src folder):  runAllModelsTest_rtx5090.bat <experiment>
REM   experiments: tableB1, figure8a, table2and3, figure9and10, figure8b, table4and5, all
REM Results are written to ..\results\<folder>; the folder names correspond to the zipped results (see Readme.md).
REM Each command restarts the Ollama server with the given batch size (-b); close the Ollama tray app before.
REM API models (Claude-Opus-4.8, GPT-5.2) require credentials and budget, see Readme.md.

set EXP=%1
if "%EXP%"=="" (
    echo usage: runAllModelsTest_rtx5090.bat [tableB1^|figure8a^|table2and3^|figure9and10^|figure8b^|table4and5^|all]
    exit /b 1
)

REM Tables 1 and B.1: all LLMs, rigid MBS (level 1+2), temperature 0 (models fitting on a 32 GB GPU + API models)
if "%EXP%"=="tableB1" goto run_tableB1
if "%EXP%"=="all" call :run_tableB1
REM Figure 8a: temperature variation, 6 LLMs (gpt-oss-20B also with thinking)
if "%EXP%"=="figure8a" goto run_figure8a
if "%EXP%"=="all" call :run_figure8a
REM Tables 2 and 3: thinking mode, 5 LLMs
if "%EXP%"=="table2and3" goto run_table2and3
if "%EXP%"=="all" call :run_table2and3
REM Figures 9 and 10: prompt variations for item selection (PVci) and code generation (PVmbs), 5 LLMs
if "%EXP%"=="figure9and10" goto run_figure9and10
if "%EXP%"=="all" call :run_figure9and10
REM Figure 8b: repeated runs (10 repetitions of 3 LLMs at temperatures 0 and 1) to assess the run-to-run variation
if "%EXP%"=="figure8b" goto run_figure8b
if "%EXP%"=="all" call :run_figure8b
REM Tables 4 and 5, Figure 11: flexible MBS (FlexuAI pipeline), 8 LLMs
if "%EXP%"=="table4and5" goto run_table4and5
if "%EXP%"=="all" call :run_table4and5
goto :eof

:run_tableB1
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Qwen3.5-35B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Qwen3.5-9B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Qwen3.5-4B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Qwen3.6-35B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n QwenCoder2.5-32B-OL -b 4 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n QwenCoder3-30B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n CodeQwen-7B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Qwen2-7B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Qwen-32B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Phi3-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Phi4-OL -b 12 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n DeepSeekCoder-16B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n codegemma-7B-OL -b 12 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n dolphincoder-15B-OL -b 12 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n nemotron-cascade-2-30B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n nemotron3-33B-OL -b 3 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n Llama3.1-8B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n glm-4.7-flash-30B-OL -b 4 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -temp 0.0 -n deepseek-r1-32B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -n Claude-Opus-4.8 -r 20 -d 44 --apiBudgetEUR 50
python modelBuilderDriver.py -f logsTM50_temp0_merge_v2 -n GPT-5.2 -r 20 -d 44 --apiBudgetEUR 50
goto :eof

:run_figure8a
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.0 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.05 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.1 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.2 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.3 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.4 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.5 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.6 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.7 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.8 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.9 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 1.0 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.0 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.05 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.1 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.2 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.3 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.4 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.5 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.6 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.7 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.8 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.9 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 1.0 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.0 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.05 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.1 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.2 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.3 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.4 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.5 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.6 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.7 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.8 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.9 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 1.0 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.0 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.05 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.1 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.2 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.3 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.4 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.5 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.6 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.7 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.8 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.9 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 1.0 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.0 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.05 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.1 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.2 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.3 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.4 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.5 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.6 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.7 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.8 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.9 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 1.0 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.0 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.05 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.1 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.2 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.3 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.4 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.5 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.6 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.7 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.8 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 0.9 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -t 1 -temp 1.0 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.0 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.05 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.1 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.15 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.2 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.3 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.4 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.5 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.6 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.7 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.8 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 0.9 -n ministral-3-14B-OL -b 16 -r 20 -d 44
python modelBuilderDriver.py -f logsTempVar_merge_v1 -temp 1.0 -n ministral-3-14B-OL -b 16 -r 20 -d 44
goto :eof

:run_table2and3
python modelBuilderDriver.py -f logsTMthk -t 1 -n gpt-oss-20B-OL -b 24 -r 20 -d 44
python modelBuilderDriver.py -f logsTMthk -t 1 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMthk -t 1 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMthk -t 1 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMthk -t 1 -n Qwen3.6-27B-OL -r 20 -d 44
goto :eof

:run_figure9and10
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 1 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 2 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 3 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 4 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 5 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 6 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 1 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 2 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 3 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 4 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 1 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 2 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 3 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 4 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 5 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 6 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 1 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 2 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 3 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 4 -n Gemma4-31B-OL -b 2 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 1 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 2 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 3 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 4 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 5 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 6 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 1 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 2 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 3 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 4 -n Gemma4-26B-OL -b 8 -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 1 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 2 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 3 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 4 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 5 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 6 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 1 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 2 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 3 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 4 -n Qwen3.5-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 1 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 2 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 3 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 4 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 5 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVci 6 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 1 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 2 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 3 -n Qwen3.6-27B-OL -r 20 -d 44
python modelBuilderDriver.py -f logsTMpv -temp 0.0 -PVmbs 4 -n Qwen3.6-27B-OL -r 20 -d 44
goto :eof

:run_figure8b
for /L %%i in (1,1,10) do (
    python modelBuilderDriver.py -f logsRepeatedRuns/logsTemp0_%%i -temp 0.0 -n Gemma4-26B-OL -b 1 -r 20 -d 44
    python modelBuilderDriver.py -f logsRepeatedRuns/logsTemp0_%%i -temp 0.0 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
    python modelBuilderDriver.py -f logsRepeatedRuns/logsTemp0_%%i -temp 0.0 -n ministral-3-14B-OL -b 1 -r 20 -d 44
    python modelBuilderDriver.py -f logsRepeatedRuns/logsTemp1_%%i -temp 1.0 -n Gemma4-26B-OL -b 8 -r 20 -d 44
    python modelBuilderDriver.py -f logsRepeatedRuns/logsTemp1_%%i -temp 1.0 -n gpt-oss-20B-OL -b 32 -r 20 -d 44
    python modelBuilderDriver.py -f logsRepeatedRuns/logsTemp1_%%i -temp 1.0 -n ministral-3-14B-OL -b 16 -r 20 -d 44
)
goto :eof

:run_table4and5
python modelBuilderDriver.py -f logsFlexuAIPipeFMBS_v4 -flexuAI -temp 0.0 -n gpt-oss-20B-OL -b 5 -r 10 -d 50
python modelBuilderDriver.py -f logsFlexuAIPipeFMBS_v4 -flexuAI -temp 0.0 -n Gemma4-31B-OL -b 10 -r 10 -d 50
python modelBuilderDriver.py -f logsFlexuAIPipeFMBS_v4 -flexuAI -temp 0.0 -n Gemma4-26B-OL -b 5 -r 10 -d 50
python modelBuilderDriver.py -f logsFlexuAIPipeFMBS_v4 -flexuAI -temp 0.0 -n ministral-3-14B-OL -b 4 -r 10 -d 50
python modelBuilderDriver.py -f logsFlexuAIPipeFMBS_v4 -flexuAI -temp 0.0 -n Qwen3.5-27B-OL -b 5 -r 10 -d 50
python modelBuilderDriver.py -f logsFlexuAIPipeFMBS_v4 -flexuAI -temp 0.0 -n Qwen3.6-27B-OL -b 5 -r 10 -d 50
python modelBuilderDriver.py -f logsFlexuAIPipeFMBS_v4 -flexuAI -t 1 -n Claude-Opus-4.8 -r 10 -d 50 --apiBudgetEUR 50
python modelBuilderDriver.py -f logsFlexuAIPipeFMBS_v4 -flexuAI -n GPT-5.2 -r 10 -d 50 --apiBudgetEUR 50
goto :eof
