# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: meshStatistics.py
#
# Details:  Mesh statistics of the flexible MBS runs (Table 5): counts how often the mesh size maxH chosen by the
#           LLM is overwritten by the recommended limit maxHLimitWithFactor; reads the debug.txt files of
#           logsFlexuAIPipeFMBS_v4 (logsFlexibleMBS_8llms_temp0_table5.zip extracted into <repo>/results)
#
# Author:   Sebastian Weyrer
# Date:     2026-10-01
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import re
import os
import glob
import math

resultsDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'logsFlexuAIPipeFMBS_v4')
FILES = sorted(glob.glob(os.path.join(resultsDir, 'log_*', 'debug.txt')), key=str.lower)

num = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
pattern = re.compile(
    r"Mode:\s*LLM code;\s*Executed code - special mesh parameters:\s*"
    r"maxHLimitWithFactor\s*=\s*" + num + r"\s*,\s*maxH\s*=\s*" + num
)
 
 
def process_file(filename):
    unequal = 0
    total = 0
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            for m in pattern.finditer(line):
                total += 1
                limit, maxh = float(m.group(1)), float(m.group(2))
                if math.isclose(limit, maxh, rel_tol=1e-12, abs_tol=1e-15):
                    continue
                unequal += 1
                if maxh < limit:
                    print(
                        f"WARNING: {os.path.basename(filename)}, line {line_no}: "
                        f"maxH ({maxh}) is smaller than "
                        f"maxHLimitWithFactor ({limit})"
                    )
    return unequal, total
 
totalPartsWithMeshStatistic = 0
overwrittenParts = 0
for filename in FILES:
    try:
        unequal, total = process_file(filename)
        print(f"{os.path.basename(os.path.dirname(filename))}: {unequal} unequal / {total} total")
        overwrittenPercent = 100-((unequal/total)*100)
        overwrittenParts += total - unequal
        totalPartsWithMeshStatistic += total
        print(f"percent of when LLM selected mesh size is overwritten: {overwrittenPercent}")
    except OSError as e:
        print(f"{os.path.basename(filename)}: ERROR ({e})")
print('mean overwritten percentage:', overwrittenParts / totalPartsWithMeshStatistic * 100)
