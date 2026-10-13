# MecEng – Benchmarking LLMs' Capabilities in Generating Rigid and Flexible Multibody Simulation Code 

This repository contains the source code accompanying the paper submitted to *Computer Methods in Applied Mechanics and Engineering* (first revision).
It implements an agentic pipeline in which large language models (LLMs) create simulation models for the multibody code [Exudyn](https://github.com/jgerstmayr/EXUDYN) from textual descriptions:

1. **Item selection** (*choose items*): the LLM selects the required Exudyn items (bodies, joints, loads, ...) from a list of helper items.
2. **Code generation** (*generate code*): using example code of the selected items, the LLM generates the Exudyn model.
3. **Evaluation**: the generated model is executed and compared with a reference solution (*sample file*), using executability, solver success, system graph comparison (topology and content) and the numerical solution.

Two pipelines are included:

* **Rigid multibody systems (MBS)**, level 1 and level 2 tasks.
* **Flexible multibody systems (FlexuAI)**, level 3 tasks: the LLM first creates the machine parts (geometry and mesh via Netgen/OCC, see `src/flexu/NetgenOCCHelper.py`) and then assembles a flexible MBS using the floating frame of reference formulation (see `src/flexu/flExudynHelper.py`).

> **Scope of this minimal version:** for the tests to be reproducible, every task requires a reference solution (sample file).
> To prevent their use in LLM training and to preserve the integrity of the benchmark, only selected reference solutions are published.
> This repository contains the sample files of the rigid MBS tasks with the IDs 1–40 (Table A.1 of the paper: all 35 level 1 tasks and the level 2 tasks `fourBarMechanismRigidBodies`, `parallelChordTrussSprings`, `prattTrussBridgeSprings`, `collidingSpheresInSpace` and `fallingMassPointContact`)
> and of the parts with the IDs 51 and 52 (`bearingBlockPart`, `unbalancedRotorPart`), together with the flexible MBS `rotorInBearingBlock` that assembles these two parts.
> The descriptions of all other tasks are still contained in `src/ExuAI/modelDescriptions.py` for documentation, but these tasks are skipped automatically when the tests are run (see the log output "tasks removed, as their sample files are not available").
> The randomized parameters of the remaining tasks are not affected by this and are identical to the runs reported in the paper.

## Repository structure

```
Readme.md
requirements.txt
src/
  modelBuilderDriver.py         command line driver for all tests (start here)
  modelBuilder.py               task processing: item selection, code generation, evaluation
  flexibleMBSBuilder.py         additional task processing for flexible MBS (FlexuAI)
  postProcessing.py             merging of results, CSV export and plots
  runAllModelsTest_rtx5090.bat  commands of all tests reported in the paper (Windows, RTX 5090)
  runAllModelsTest_h100.bash    commands of all tests reported in the paper (Linux, H100; large LLMs)
  ExuAI/
    agent.py                    LLM agent (prompts, execution and evaluation of code)
    myLLM.py                    unified interface to the LLM backends and list of LLMs (llmModelsDict)
    modelDescriptions.py        task descriptions with parameter ranges (rigid MBS, parts, flexible MBS)
    modelLoader.py              loading and randomization of task descriptions
    exudynHelper.py             helper file with Exudyn items (examples given to the LLM)
    templatesAndLists.py        prompt templates (incl. prompt variations)
    compareGraphs.py            system graph comparison and error categorization
    flexUtilities.py            utilities for meshing and FFRF (flexible bodies)
    readHelperFile.py, utilities.py, logger.py
  sampleFiles/                  reference solutions of the rigid MBS tasks
  flexu/
    flExudynHelper.py           helper file for flexible MBS creation
    NetgenOCCHelper.py          helper file for part (geometry/mesh) creation
    sampleFiles/                reference solutions of the flexible MBS task and its parts
misc/
  flexuAI/                      scripts for paper tables/figures and FFRF reference examples
  paperTableData/               merged result workbooks used for the paper tables
  visualization/                example models used for the task illustrations in the paper
```

## Installation

The tests of the paper were run on Windows (NVIDIA RTX 5090, 32 GB) and Linux (NVIDIA H100). The pipeline was tested with Python P3.11-3.13 on Linux and Windows.

1. Install the Python packages:
   ```
   pip install -r requirements.txt
   ```
   The code requires Exudyn ≥ 1.10.150; the public release Exudyn 1.11.0 (PyPI) works.
   The paper results were computed with the development version Exudyn 1.10.160.dev1, which is not available on PyPI;
   minor differences of results with other Exudyn versions cannot be excluded.
2. Install the [Ollama](https://ollama.com) server (the results were computed with Ollama 0.24.0). Pull the LLMs you want to test, e.g.
   ```
   ollama pull gpt-oss:20b
   ```
   If a model is not available locally, the driver tries to pull it automatically.
   The mapping between the short names used on the command line and the Ollama tags is given in `llmModelsDict` in `src/ExuAI/myLLM.py`.
3. Optional, for the proprietary models:
   * `Claude-Opus-4.8`: `pip install anthropic` and set the environment variable `ANTHROPIC_API_KEY`.
   * `GPT-5.2`: accessed via the *Academic AI* service of the University of Innsbruck (OpenAI-compatible API, not publicly accessible).
     Set `CLIENT_ID` and `CLIENT_SECRET` and, if needed, `--apiBaseURL` and `--apiAuthMethod`.
   * The option `--apiBudgetEUR` sets a cost limit for API runs.
4. Optional, other backends (not used in the paper): Hugging Face transformers (`torch`, `transformers`, `accelerate`, `bitsandbytes`), vLLM, GPT4All and llama.cpp; see the header of `src/ExuAI/myLLM.py`.

## Running a test

All commands are run from the `src` folder. A single test is run with `modelBuilderDriver.py`, e.g. 20 randomizations of all rigid MBS tasks with gpt-oss-20B at temperature 0 and batch size 32:

```
cd src
python modelBuilderDriver.py -n gpt-oss-20B-OL -temp 0.0 -b 32 -r 20 -d 44 -f myTest
```

Important options (`python modelBuilderDriver.py -h` lists all options and the available LLM short names):

| option | meaning |
|---|---|
| `-n` | LLM short name (see `llmModelsDict` in `src/ExuAI/myLLM.py`) |
| `-r` | number of randomized parameter sets per task |
| `-d` | maximum difficulty of tasks; `44` = all rigid MBS tasks, `50` = all flexible MBS tasks |
| `-temp` | sampling temperature (default: LLM default) |
| `-t` | thinking mode (0 = off/low, 1 = on/medium, 2 = high) |
| `-b` | batch size, i.e. number of parallel requests (`OLLAMA_NUM_PARALLEL`); the driver restarts the Ollama server with this setting |
| `-PVci`, `-PVmbs` | prompt variation for item selection (1–6) and code generation (1–4) |
| `-flexuAI` | run the flexible MBS pipeline (level 3 tasks) |
| `-f` | results folder (written to `results/<folder>` in the repository root) |
| `-m` | only merge existing results of a results folder (CSV export and plots) |

For Ollama batching on Windows, close the Ollama tray application before running the tests; the driver starts its own Ollama server.

Each run writes a folder `results/<folder>/log_<LLM>[-B<batch>][-Thk<thinking>][-T<temperature>][-PVCI<i>|-PVMBS<i>]` containing
`results.json` (all prompts, responses and evaluation data), `humanReadableLog.md`, `quickinfo.md`, `debug.txt`, the generated (`*LLM`) and reference (`*Sample`) models.
After each run, all results of the results folder are merged into `allLLMresults.json`, `overallResults.csv` (one line per LLM run) and `overallResultsSuccessRates.csv` (success rates per task).

## Reproducing the tests of the paper

The scripts `src/runAllModelsTest_rtx5090.bat` (Windows) and `src/runAllModelsTest_h100.bash` (Linux) contain the commands of all tests reported in the paper.
Each experiment is selected by an argument, e.g. `runAllModelsTest_rtx5090.bat figure8a` or `bash runAllModelsTest_h100.bash tableB1_h100`.
The LLMs larger than 32 GB (Tables 1 and B.1) were run on the H100 (`tableB1_h100`), all other tests on the RTX 5090.

| experiment (script argument) | paper | results folder |
|---|---|---|
| `tableB1`, `tableB1_h100` | Tables 1 and B.1, Figure 7 (all LLMs, temperature 0) | `logsTM50_temp0_merge_v2` |
| `figure8a` | Figure 8a (temperature variation, 6 LLMs) | `logsTempVar_merge_v1` |
| `figure8b` | Figure 8b (repeated runs: 10 repetitions of gemma4:26b, gpt-oss:20b and ministral-3:14b at temperatures 0 and 1) | `logsRepeatedRuns/logsTemp0_1` … `logsTemp1_10` |
| `table2and3` | Tables 2 and 3 (thinking mode, 5 LLMs) | `logsTMthk` |
| `figure9and10` | Figures 9 and 10 (prompt variations, 5 LLMs) | `logsTMpv` |
| `table4and5` | Tables 4 and 5, Figure 11 (flexible MBS, 8 LLMs) | `logsFlexuAIPipeFMBS_v4` |

In Table 1, gemma4:26b is reported with its temperature-0 run of the temperature study (`logsTempVar_merge_v1`), which is also the reference run of its repetitions in Figure 8b.

The result folders are created in `results/` in the repository root and have the same names as the result folders of the paper, so the post-processing below can be applied directly.
Note that LLM inference is not fully deterministic: even at temperature 0, the overall success rate of repeated runs varies by up to about 0.5 percentage points (standard deviation, Figure 8b), e.g. due to batching on the GPU; the batch size also slightly affects the results.
Together with the fact that this minimal version contains only a subset of the tasks (see above), rerunning the tests reproduces the reported results only statistically and for the included tasks.

### Results data

The raw results of the paper (prompts, LLM responses, generated models and evaluations) are not published, as they contain the reference solutions of all tasks, including the withheld ones.
All results can be regenerated for the published tasks with the scripts above.

### Post-processing

* Merge a results folder and create CSV files and plots (`-time`: success vs. release date (Figure 7), `-temperature`: success vs. temperature (Figure 8a), `-pv`: prompt variation bar charts (Figures 9 and 10)):
  ```
  cd src
  python postProcessing.py ../results/logsTempVar_merge_v1 -temperature
  python postProcessing.py ../results/logsTMpv -pv
  ```
* Statistics of the repeated runs (Figure 8b): mean, standard deviation and range of the overall success per LLM and temperature, compared with the reference runs in `results/logsTM50_temp0_merge_v2` and `results/logsTempVar_merge_v1` (if available); writes `results/repeatedRunsStatistics.pdf` and `.csv`:
  ```
  cd src
  python postProcessing.py ../results -repeat
  ```
* LaTeX tables and figures of the paper (written to `misc/paperOutput`):
  * `misc/flexuAI/generateModelComparisonTable.py`: LLM comparison tables (Tables 1 and B.1), from `misc/paperTableData/overallResults50models_2026-06-25_mergedLLMs.xlsx` (merged results of all tasks of the paper)
  * `misc/flexuAI/generatePaperPlotsAndTables.py`: thinking vs. no thinking tables (Tables 2 and 3), from `misc/paperTableData/overallResultsThinkingFinal.xlsx` (merged results of all tasks of the paper)
  * `misc/flexuAI/flExudynPostProcessing.py`: flexible MBS tables and figures (Tables 4 and 5, Figure 11), from `results/logsFlexuAIPipeFMBS_v4/overallResults.csv` and `overallResultsSuccessRates.csv`
  * `misc/flexuAI/meshStatistics.py`: statistics of the mesh sizes chosen by the LLMs, from `results/logsFlexuAIPipeFMBS_v4` (printed to the console only, no output file)
* `misc/visualization`: example models for the illustrations of the tasks in the paper. The scripts open Exudyn's OpenGL renderer (a display is required) and must be run from this folder.
  The part scripts (e.g. `windTurbineTowerPart.py`, `crankPartAdvanced.py`) write `.npz` files that are loaded by the assembly scripts (`level3bTaskExample.py`, `flexibleSliderCrankAdvanced3.py`);
  `rotorInBearingBlock.py` loads `bearingBlockPart.npz` and `unbalancedRotorPart.npz`, which are written by running the corresponding sample files in `src/flexu/sampleFiles` and copying them into `misc/visualization`.

## Authors

* Johannes Gerstmayr (Department of Mechatronics, University of Innsbruck), corresponding author: johannes.gerstmayr@uibk.ac.at
* Sebastian Weyrer (Department of Mechatronics, University of Innsbruck)
* Tobias Möltner (Department of Mechatronics, University of Innsbruck)
* Peter Manzl (Institute of Materials Resource Management, University of Augsburg)
* Michael Pieber (Department of Mechatronics, University of Innsbruck)

## Citation

If you use this code, please cite the paper:

```
J. Gerstmayr, S. Weyrer, T. Möltner, P. Manzl, M. Pieber:
Large Language Models and their Awareness of Mechanics and Spatial Geometry.
Computer Methods in Applied Mechanics and Engineering (submitted).
```

## License

See the headers of the source files and `LICENSE`.
