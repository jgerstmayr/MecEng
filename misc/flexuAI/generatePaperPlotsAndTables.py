# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: generatePaperPlotsAndTables.py
#
# Details:  Reads the merged thinking/no-thinking results from
#           misc/paperTableData/overallResultsThinkingFinal.xlsx and exports
#           paper-ready LaTeX table snippets (.tex) into
#           misc/paperOutput.
#
#           Generated artifacts:
#             - tableOverallResultsThinking.tex : per-model summary table
#                 (choose components / overall success / total tokens / LLM dur.,
#                  each split into thinking vs. no-thinking columns)
#             - tableErrorCategoriesThinking.tex : graph-content error-category
#                 table with proportion (%) and average count, each split into
#                 thinking vs. no-thinking columns
#
# Authors:  Michael Pieber, Claude Opus4.8
# Date:     2026-06-29
# Notes:    Standalone export script (analogous to ExuAI/modelDescriptions.py
#           __main__). Run from anywhere:
#               python misc/flexuAI/generatePaperPlotsAndTables.py
#
# License:  BSD-3 license
#
# Installation:
#           pip install openpyxl
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import openpyxl
from generateModelComparisonTable import modelFootnotes

#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# configuration
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# resolve paths relative to this file: <repo>/misc/flexuAI/<thisfile>
scriptDir = os.path.dirname(os.path.abspath(__file__))
miscDir = os.path.dirname(scriptDir)
paperTableDataDir = os.path.join(miscDir, 'paperTableData')
xlsxPath = os.path.join(paperTableDataDir, 'overallResultsThinkingFinal.xlsx')
outputDir = os.path.join(miscDir, 'paperOutput')

# block marker labels in column A of the workbook
blockThinking = 'thinking'
blockNoThinking = 'noThinking'
blockMarkers = {
    'with thinking': blockThinking,
    'without thinking': blockNoThinking,
}

# workbook column header -> internal key (only the columns we actually use)
columnKeys = {
    'LLM full name': 'modelName',
    'LLM dur. (min)': 'durationMin',
    'total tokens': 'totalTokensK',
    'choose items': 'chooseComponents',
    'overall success': 'overallSuccess',
}

# error-category columns: display name (paper wording, lowercased) -> workbook header
# order matches the column order in the source workbook
errorCategories = {
    'kinematics / marker parameters': 'kin./marker nERR',
    'rigid body configuration': 'RB ref.conf. nERR',
    'initial velocities': 'vel. nERR',
    'inertia and mass parameters': 'inertia/mass nERRs',
    'gravity and load parameters': 'gravity/load nERR',
    'connector / constraint parameters': 'connect./constr. nERR',
    'other': 'other nERR',
}


#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# workbook parsing
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def ParseThinkingWorkbook(workbookPath):
    """
    @brief Parse the merged thinking/no-thinking results workbook.
    @param workbookPath absolute path to overallResultsThinkingFinal.xlsx
    @return tuple (models, modelOrder) where models is a dict
            models[modelName][block][internalKey] = value, block in
            {blockThinking, blockNoThinking}, and modelOrder is the model name
            order as it first appears in the thinking block.
    """
    workbook = openpyxl.load_workbook(workbookPath, data_only=True)
    worksheet = workbook[workbook.sheetnames[0]]

    rows = [list(r) for r in worksheet.iter_rows(values_only=True)]

    models = {}
    modelOrder = []
    currentBlock = None
    headerMap = None  # internalKey -> column index for the current block

    for row in rows:
        firstCell = row[0] if row else None
        firstText = str(firstCell).strip().lower() if firstCell is not None else ''

        # detect block marker rows (e.g. "with thinking:")
        matchedMarker = next((b for marker, b in blockMarkers.items()
                              if firstText.startswith(marker)), None)
        if matchedMarker is not None:
            currentBlock = matchedMarker
            headerMap = None
            continue

        # detect the header row that starts each block
        if firstCell == 'Case Name':
            headerMap = {}
            for colIndex, headerName in enumerate(row):
                if headerName in columnKeys:
                    headerMap[columnKeys[headerName]] = colIndex
                if headerName in errorCategories.values():
                    headerMap[headerName] = colIndex
            continue

        # data rows: require an active block, a header, and a model name cell
        if currentBlock is None or headerMap is None:
            continue
        modelNameIndex = headerMap.get('modelName')
        if modelNameIndex is None:
            continue
        modelName = row[modelNameIndex]
        if not modelName or not str(modelName).strip():
            continue  # blank / average / reduction rows end the data block

        record = {}
        for internalKey, colIndex in headerMap.items():
            record[internalKey] = row[colIndex]

        if modelName not in models:
            models[modelName] = {}
        models[modelName][currentBlock] = record

        if currentBlock == blockThinking and modelName not in modelOrder:
            modelOrder.append(modelName)

    # fall back to any ordering if the thinking block was empty
    if not modelOrder:
        modelOrder = list(models.keys())

    return models, modelOrder


def ComputeErrorStats(models, modelOrder, block):
    """
    @brief Aggregate graph-content error counts over all models for a single
           block and return both the per-category average count and the
           per-category proportion (% of all errors).
    @param models parsed model dict from ParseThinkingWorkbook
    @param modelOrder ordered list of model names to aggregate
    @param block blockThinking or blockNoThinking
    @return tuple (averages, proportions), each a dict displayName -> value;
            averages are mean error counts per model, proportions are in
            percent (0..100) and sum to ~100 within the block.
    """
    categorySums = {displayName: 0.0 for displayName in errorCategories}
    numberOfModels = 0
    for modelName in modelOrder:
        record = models[modelName].get(block, {})
        if not record:
            continue
        numberOfModels += 1
        for displayName, header in errorCategories.items():
            value = record.get(header)
            if isinstance(value, (int, float)):
                categorySums[displayName] += float(value)

    totalErrors = sum(categorySums.values())
    averages = {displayName: (categorySums[displayName] / numberOfModels
                              if numberOfModels > 0 else 0.0)
                for displayName in errorCategories}
    proportions = {displayName: (100.0 * categorySums[displayName] / totalErrors
                                 if totalErrors > 0.0 else 0.0)
                   for displayName in errorCategories}
    return averages, proportions


#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# LaTeX table export
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def FormatPercent(value, info=None):
    """@brief Format a fraction (0..1) as a LaTeX percent string, e.g. 0.998 -> '99.8\\%'."""
    if not isinstance(value, (int, float)):
        return '--'
    return f'{100.0 * value:.1f}\\%'

def SpecialFormatRate(value, info=None, columnValues=None):
    if not isinstance(value, (int, float)):
        return '--'
    if info == 1 or info == 2 or info == 6:
        if info == 6:
            previousValue = 3
        else:
            previousValue = info - 1
        return f'{100.0 * value:.1f} ({100.0 * value / columnValues[previousValue]:.1f})'
    return f'{100.0 * value:.1f}'


def FormatInt(value):
    """@brief Format a numeric value as a rounded integer string."""
    if not isinstance(value, (int, float)):
        return '--'
    return f'{round(value):d}'
    

def WriteSummaryTable(models, modelOrder, outputPath):
    """
    @brief Write the per-model summary table (choose components / overall success /
           total tokens / LLM dur.), each split into thinking vs. no-thinking.
    @param models parsed model dict
    @param modelOrder ordered list of model names (rows)
    @param outputPath absolute path of the .tex file to write
    """
    lines = []
    lines.append('% auto-generated by misc/flexuAI/generatePaperPlotsAndTables.py')
    lines.append('% requires \\usepackage{booktabs} in preamble')
    lines.append('\\begin{tabular}{l cc cc cc cc}')
    lines.append('\\toprule')
    lines.append(' & \\multicolumn{2}{c}{\\textbf{choose comp.}} & '
                 '\\multicolumn{2}{c}{\\textbf{overall success}} & '
                 '\\multicolumn{2}{c}{\\textbf{total tokens (k)}} & '
                 '\\multicolumn{2}{c}{\\textbf{LLM dur. (min.)}} \\\\')
    lines.append('\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}'
                 '\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}')
    lines.append('\\textbf{LLM} & thk. & no-thk. & thk. & no-thk. & '
                 'thk. & no-thk. & thk. & no-thk. \\\\')
    lines.append('\\midrule')

    for modelName in modelOrder:
        thk = models[modelName].get(blockThinking, {})
        no = models[modelName].get(blockNoThinking, {})
        cells = [
            str(modelName),
            FormatPercent(thk.get('chooseComponents')),
            FormatPercent(no.get('chooseComponents')),
            FormatPercent(thk.get('overallSuccess')),
            FormatPercent(no.get('overallSuccess')),
            FormatInt(thk.get('totalTokensK')),
            FormatInt(no.get('totalTokensK')),
            FormatInt(thk.get('durationMin')),
            FormatInt(no.get('durationMin')),
        ]
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')

    with open(outputPath, 'w', encoding='utf-8') as fileHandle:
        fileHandle.write('\n'.join(lines) + '\n')
    print(f'Summary table written to: {outputPath}')

def WriteGroupedSuccessRateTable(rowGroups, columnHeaders, outputPath,
                                 formatter=None, spanningHeader='success rate (\\%)'):
    """
    @brief Write a LaTeX table with a spanning header over the metric columns,
           multiple row-groups separated by \\midrule (e.g. proprietary vs.
           open-weight models), and an automatic block of mean rows per group.
    @param rowGroups list of dicts, each with:
             'rows': list of (label, values) tuples, values aligned with columnHeaders
             'meanLabel': label for that group's mean row (e.g. 'Mean (proprietary)')
           Groups are written in order, separated by \\midrule. After all data
           rows, a final \\midrule-separated block lists each group's mean row,
           in the same order.
    @param columnHeaders list of metric column headers (e.g. ['items', 'solver', ...])
    @param outputPath absolute path of the .tex file to write
    @param formatter function(value) -> str applied to every cell (default: FormatPercent)
    @param spanningHeader text shown above the metric columns (set to None to omit)
    """
    formatter = formatter or FormatPercent
    numberOfColumns = len(columnHeaders)
    lines = []
    lines.append('% auto-generated by misc/flexuAI/generatePaperPlotsAndTables.py')
    lines.append('% requires \\usepackage{booktabs} in preamble')
    if spanningHeader == 'success rate (\\%)':
        lines.append('\\begin{tabular}{l' + ' ccc|c|' + 'c' * (numberOfColumns-4) + '}')
    else:
        lines.append('\\begin{tabular}{l' + 'c' * (numberOfColumns) + '}')
    lines.append('\\toprule')

    if spanningHeader is not None:
        lines.append(f' & \\multicolumn{{{numberOfColumns}}}{{c}}{{\\textbf{{{spanningHeader}}}}} \\\\')
        lines.append(f'\\cmidrule(lr){{2-{numberOfColumns + 1}}}')

    lines.append('\\textbf{LLM} & ' + ' & '.join(columnHeaders) + ' \\\\')
    lines.append('\\midrule')

    # data rows, grouped, with \midrule between groups
    for groupIndex, group in enumerate(rowGroups):
        if groupIndex > 0:
            lines.append('\\midrule')
        for label, values in group['rows']:
            if label in modelFootnotes:
                label += modelFootnotes[label]
            if formatter == SpecialFormatRate:
                rowList = []
                for i, value in enumerate(values):
                    rowList += [SpecialFormatRate(value, info=i, columnValues=values)]
                cells = [str(label)] + rowList
            else:
                cells = [str(label)] + [formatter(v, g) for (v, g) in zip(values, columnHeaders)]
            lines.append(' & '.join(cells) + ' \\\\')

    # mean-row block
    lines.append('\\midrule')
    for group in rowGroups:
        meanValues = ComputeColumnMeans([values for _, values in group['rows']])
        if formatter == SpecialFormatRate:
            rowList = []
            for i, value in enumerate(meanValues):
                rowList += [SpecialFormatRate(value, info=i, columnValues=meanValues)]
            cells = [group['meanLabel']] + rowList
        else:
            cells = [group['meanLabel']] + [formatter(v, g) for (v, g) in zip(meanValues, columnHeaders)]
        lines.append(' & '.join(cells) + ' \\\\')

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')

    with open(outputPath, 'w', encoding='utf-8') as fileHandle:
        fileHandle.write('\n'.join(lines) + '\n')
    print(f'Table written to: {outputPath}')


def ComputeColumnMeans(rowsOfValues):
    """
    @brief Compute the per-column mean over a list of value-rows, ignoring
           non-numeric entries column-wise.
    @param rowsOfValues list of lists/tuples of numeric values, all same length
    @return list of column means (float), same length as each row
    """
    if not rowsOfValues:
        return []
    numberOfColumns = len(rowsOfValues[0])
    means = []
    for columnIndex in range(numberOfColumns):
        columnValues = [row[columnIndex] for row in rowsOfValues
                        if isinstance(row[columnIndex], (int, float))]
        means.append(sum(columnValues) / len(columnValues) if columnValues else float('nan'))
    return means

def WriteErrorCategoryTable(averagesThinking, averagesNoThinking,
                            proportionsThinking, proportionsNoThinking,
                            categoryOrder, outputPath):
    """
    @brief Write the graph-content error-category table with proportion (%) and
           average count, each split into thinking vs. no-thinking columns.
    @param averagesThinking dict displayName -> mean error count (thinking)
    @param averagesNoThinking dict displayName -> mean error count (no-thinking)
    @param proportionsThinking dict displayName -> percent (thinking)
    @param proportionsNoThinking dict displayName -> percent (no-thinking)
    @param categoryOrder ordered list of category display names (rows)
    @param outputPath absolute path of the .tex file to write
    """
    lines = []
    lines.append('% auto-generated by misc/flexuAI/generatePaperPlotsAndTables.py')
    lines.append('% requires \\usepackage{booktabs} in preamble')
    lines.append('\\begin{tabular}{l cc cc}')
    lines.append('\\toprule')
    lines.append(' & \\multicolumn{2}{c}{\\textbf{proportion (\\%)}} & '
                 '\\multicolumn{2}{c}{\\textbf{average count}} \\\\')
    lines.append('\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}')
    lines.append('\\textbf{error category} & thk. & no-thk. & '
                 'thk. & no-thk. \\\\')
    lines.append('\\midrule')
    for displayName in categoryOrder:
        latexName = displayName.replace('&', '\\&')
        lines.append(f'{latexName} & '
                     f'{proportionsThinking[displayName]:.1f} & '
                     f'{proportionsNoThinking[displayName]:.1f} & '
                     f'{averagesThinking[displayName]:.1f} & '
                     f'{averagesNoThinking[displayName]:.1f} \\\\')
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')

    with open(outputPath, 'w', encoding='utf-8') as fileHandle:
        fileHandle.write('\n'.join(lines) + '\n')
    print(f'Error-category table written to: {outputPath}')


#%%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# main
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Main():
    """@brief Parse the workbook and export both LaTeX tables."""
    if not os.path.isfile(xlsxPath):
        raise FileNotFoundError(f'input workbook not found: {xlsxPath}')
    os.makedirs(outputDir, exist_ok=True)

    models, modelOrder = ParseThinkingWorkbook(xlsxPath)
    print(f'Parsed {len(modelOrder)} models: {", ".join(str(m) for m in modelOrder)}')

    averagesThinking, proportionsThinking = ComputeErrorStats(models, modelOrder, blockThinking)
    averagesNoThinking, proportionsNoThinking = ComputeErrorStats(models, modelOrder, blockNoThinking)

    # category order follows the source-workbook column order
    categoryOrder = list(errorCategories.keys())

    WriteSummaryTable(models, modelOrder,
                      os.path.join(outputDir, 'tableOverallResultsThinking.tex'))
    WriteErrorCategoryTable(averagesThinking, averagesNoThinking,
                            proportionsThinking, proportionsNoThinking, categoryOrder,
                            os.path.join(outputDir, 'tableErrorCategoriesThinking.tex'))

    print(f'\nAll tables written to: {outputDir}')


if __name__ == '__main__':
    Main()
