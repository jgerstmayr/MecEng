# -*- coding: utf-8 -*-
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This is part of the AI engineering lab project
#
# Filename: logger.py
#
# Details:  This script defines a class for logging different information of agent-functionalities.
#           It logs the information both in human- and machinereadable format depending on the information provided.
#
# Authors:  Tobias Möltner, Johannes Gerstmayr
# Date:     2024-12-20
#
# License:  BSD-3 license
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import atexit
import os
import time
from datetime import datetime
try: 
    from .utilities import JoinPath, IsScalar
except: 
    from utilities import JoinPath, IsScalar


#set up logger
#debugLogger: logs everything, incl. time stamps, using logging package
#humanLogger: creates human readable file
class Logger:
    def __init__(self, logDir, suffix='', appendMode=False):
        """@brief Structured logger writing humanReadableLog/quickinfo/debug files into logDir.
           @param logDir directory the three log files are written to
           @param suffix optional suffix appended to each log file name
           @param appendMode keep and append to existing log files instead of clearing them.
                  Needed when a run is split across processes: the vLLM backend cannot free
                  VRAM in-process, so Stage 0/1 and Stage 2/3 run as separate invocations
                  against the SAME log directory, and the second one would otherwise wipe the
                  first one's logs (including all Stage-1 timings) on startup.
        """
        self.limitConsoleText = 120 #can be adapted from outside
        self.errorCount = 0 #increased with LogError
        self.warningCount = 0 #increased with LogWarning
        self.appendMode = appendMode

        #! Write buffering. Every log call used to be its own open/write/flush/close, and on
        #! /mnt/... under WSL each of those costs ~12.7 ms against 0.004 ms on ext4 -- a 3000x
        #! penalty paid per CALL, not per byte. One LogText(separator=True) fans out into five
        #! or six such calls, so a sweep run spent ~17 min writing its 14 MB of logs with the
        #! GPU idle throughout (measured 2026-08-29: the log grew at a write-bound 14 KB/s
        #! while utilisation sat at 0%). Batching by size or age turns thousands of round-trips
        #! into a handful. A hard kill loses at most the last flush interval, and nothing that
        #! matters for provenance lives only in the log.
        self.writeBuffers = {}          #! fileName -> list of pending strings
        self.writeBufferBytes = {}      #! fileName -> pending byte count
        self.lastFlushTime = {}         #! fileName -> time.time() of last flush
        self.flushMaxBytes = 65536      #! flush once a file has this much pending
        self.flushMaxSeconds = 2.0      #! ...or this long since its last flush
        atexit.register(self.FlushAll)

        def Clear_log_file(fileName):
            """Clear the log file when the logger is initialized, unless appending."""
            if appendMode:
                #! mark the boundary so a split run is still readable as one timeline
                if os.path.exists(fileName):
                    with open(fileName, 'a', encoding='utf-8') as file:
                        file.write('\n' + '='*39 + '\n'
                                   '=== log continues in a new process ===\n'
                                   + '='*39 + '\n')
                    return
            with open(fileName, 'w', encoding='utf-8') as file:
                file.write("")  # Overwrite with an empty file
    
        def ResetLogger(logger):
            # Reset logger if needed
            if logger.hasHandlers():
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                    handler.close()
        
        if len(logDir) > 0 and logDir[-1] != '/':
            logDir += '/'
        self.logDir = logDir

        humanLogFile = JoinPath(logDir,'humanReadableLog'+suffix+'.md')
        infoLogFile = JoinPath(logDir,'quickinfo'+suffix+'.md')
        debugFile = JoinPath(logDir,'debug'+suffix+'.txt')
        
        #create directory if it does not exist
        try:
            os.makedirs(os.path.dirname(humanLogFile), exist_ok=True)
        except:
            pass #makedirs may fail on some systems, but we keep going
        
        self.humanLoggerFileName = humanLogFile
        Clear_log_file(humanLogFile)
        self.infoLoggerFileName = infoLogFile
        Clear_log_file(infoLogFile)
        self.debugLoggerFileName = debugFile
        Clear_log_file(debugFile)

    def ResetCounters(self):
        self.errorCount = 0
        self.warningCount = 0
        

    def WriteFile(self, fileName, text, newLine=True, nRetries=5, retryDelaySeconds=0.2):
        """@brief Append text to a log file, tolerating transient lock failures.

           Log writes must never abort a run. On Windows-backed storage (and especially
           /mnt/... under WSL, where writes cross the 9p boundary) another process holding
           the file open without share-write makes the write fail with PermissionError --
           antivirus and file-sync clients do this routinely. Observed 2026-08-29: a single
           denied write to debug.txt killed a completed 115-task Stage-1a run, losing ~5
           minutes of GPU work purely because a log line could not be appended.
           After the retries are exhausted the text is dropped and a note is printed to the
           console: losing a log line is always preferable to losing the run.
           @param nRetries attempts after the first failure, with a short delay between them
           @param retryDelaySeconds delay between attempts; locks of this kind clear in ms
        """
        pending = str(text) + '\n'*newLine
        self.writeBuffers.setdefault(fileName, []).append(pending)
        self.writeBufferBytes[fileName] = self.writeBufferBytes.get(fileName, 0) + len(pending)
        self.lastFlushTime.setdefault(fileName, time.time())
        if (self.writeBufferBytes[fileName] >= self.flushMaxBytes
                or time.time() - self.lastFlushTime[fileName] >= self.flushMaxSeconds):
            self.FlushFile(fileName, nRetries=nRetries, retryDelaySeconds=retryDelaySeconds)


    def FlushFile(self, fileName, nRetries=5, retryDelaySeconds=0.2):
        """@brief Append everything buffered for one file, tolerating transient lock failures.
           @param nRetries attempts after the first failure, with a short delay between them
           @param retryDelaySeconds delay between attempts; locks of this kind clear in ms
        """
        buffered = self.writeBuffers.get(fileName)
        if not buffered:
            return
        payload = ''.join(buffered)
        #! cleared before the write: a failing flush must not requeue the same text forever
        self.writeBuffers[fileName] = []
        self.writeBufferBytes[fileName] = 0
        self.lastFlushTime[fileName] = time.time()
        for attemptIndex in range(nRetries + 1):
            try:
                with open(fileName, 'a', encoding='utf-8') as file:
                    file.write(payload)  # Append mode
                    file.flush()
                return
            except (PermissionError, OSError) as exceptionInstance:
                if attemptIndex >= nRetries:
                    print(f"Logger WARNING: could not write to {fileName} after "
                          f"{nRetries+1} attempts ({type(exceptionInstance).__name__}: "
                          f"{exceptionInstance}); dropping {len(payload)} buffered bytes "
                          f"and continuing.")
                    return
                time.sleep(retryDelaySeconds)


    def FlushAll(self):
        """@brief Flush every buffered log file; registered with atexit, safe to call twice."""
        for fileName in list(self.writeBuffers.keys()):
            try:
                self.FlushFile(fileName)
            except Exception:  #! a log flush must never raise during interpreter shutdown
                pass


    def GetTimeStamp(self):
        # Get the current timestamp
        timestamp = datetime.now()
        
        # Extract milliseconds properly
        milliseconds = timestamp.microsecond // 1000  # Convert microseconds to milliseconds
        
        # Format the timestamp correctly
        timestamp_str = timestamp.strftime(f"%Y-%m-%d %H:%M:%S.{milliseconds:03d}")
        
        return timestamp_str
    

    #write debug messages
    def LogDebug(self, debugMessage, printToConsole=False, separator=False):
        if separator:
            self.WriteFile(self.debugLoggerFileName, '\n=======================================\n')
        debugMessage = str(debugMessage)
        self.WriteFile(self.debugLoggerFileName, self.GetTimeStamp() + ' - '+debugMessage)
        if printToConsole:
            print(debugMessage)

    #write plain text to debug, quick and human readable
    def LogText(self, text, quickinfo=False, timestamp=True, style=0, separator=False, 
                newLine=True, printToConsole=False, limitConsoleText=-1):
        timeStr = self.GetTimeStamp()
        text = str(text) #if None, etc. => string operations would not work
        
        if separator:
            self.WriteFile(self.humanLoggerFileName, '\n***\n')
            if quickinfo:
                self.WriteFile(self.infoLoggerFileName, '\n***\n')
            self.WriteFile(self.debugLoggerFileName, '\n=======================================')

        if timestamp:
            self.WriteFile(self.debugLoggerFileName, '\n'+timeStr+':')

        if timestamp == 2:
            self.WriteFile(self.humanLoggerFileName, '\n*time stamp: '+timeStr+'*\n')
            if quickinfo:
                self.WriteFile(self.infoLoggerFileName, '\n*time stamp: '+timeStr+'*\n')

        self.WriteFile(self.debugLoggerFileName, text, newLine=newLine)
        
        
        if printToConsole:
            textConsole = text
            if limitConsoleText > 0 and len(text)>limitConsoleText:
                textConsole = text[:limitConsoleText]+'...'
            print(textConsole)


        if style == STYLE.bold:
            text = '**'+text.strip()+'** '
        elif style == STYLE.italics:
            text = '*'+text.strip()+'* '
        elif style == STYLE.python:
            copyText = text.replace('```python','').replace('```','')
            text = '\n```\n'+copyText+'\n```\n'

        self.WriteFile(self.humanLoggerFileName, text, newLine=newLine)
        if quickinfo:
            self.WriteFile(self.infoLoggerFileName, text, newLine=newLine)


    def LogError(self, errorMessage, printToConsole=True):
        self.errorCount += 1
        errorMessage = str(errorMessage)
        self.LogText('***\nERROR:', style=STYLE.bold, printToConsole=False, separator=True, quickinfo=True)
        self.LogText(errorMessage+'\n***', style=STYLE.python, printToConsole=False, quickinfo=True,
                     limitConsoleText=self.limitConsoleText, timestamp=False)
        if printToConsole:
            print('*** ERROR: '+errorMessage)
        
    def LogWarning(self, warningMessage, printToConsole=False):
        self.warningCount += 1
        warningMessage = str(warningMessage)
        self.LogText('WARNING:', style=STYLE.bold, printToConsole=printToConsole, separator=True)
        self.LogText(warningMessage, style=STYLE.python, printToConsole=printToConsole, timestamp=False)
        

    #log dictionary as list
    def LogDict(self, infoDict, inline=False, header='', timestamp=True, quickinfo=False, excludeCodeLLM=False,
                level1Inline=None, separator=False):
        
        if header != '':
            self.LogText(header+':'*(inline), style=STYLE.bold, separator=separator, timestamp=timestamp, quickinfo=quickinfo, 
                         newLine=(not inline))

        if level1Inline is None:
            level1Inline = inline

        itemSymbol = ' - '
        endLine = '\n\n'
        if inline:
            itemSymbol = ''
            endLine = ''
        endLine2 = endLine
        if level1Inline:
            endLine2 = ''
        
        for key, value in infoDict.items():
            if type(value) == dict:
                self.LogText(itemSymbol+key+': '+endLine2, timestamp=False, 
                             newLine=False, quickinfo=quickinfo)
                itemSymbol1 = '   + '
                if level1Inline:
                    itemSymbol1 = '['
                for key1, value1 in value.items():
                    if excludeCodeLLM:
                        if ('code' in key1
                            or 'LLM prompt' in key1
                            or 'LLM response' in key1
                            ): continue

                    sValue1 = '"'+value1+'"' if type(value1)==str else str(value1)
                    self.LogText(itemSymbol1+key1+': '+sValue1+endLine2, timestamp=False, 
                                 newLine=False, quickinfo=quickinfo)
                    if level1Inline: 
                        itemSymbol1 = ', '
                if level1Inline: 
                    self.LogText(']'+'\n\n'*(not inline), timestamp=False, newLine=False, quickinfo=quickinfo)
            else:
                if excludeCodeLLM:
                    if ('general exudyn model' in key 
                        or key.startswith('evaluation model')
                        or 'LLM prompt' in key 
                        or 'LLM response' in key 
                        ):
                        continue
                sValue = '"'+value+'"' if type(value)==str else str(value)
                self.LogText(itemSymbol+key+': '+sValue+endLine, timestamp=False, newLine=False, quickinfo=quickinfo)
            if inline: itemSymbol = ', '
        
        if inline or level1Inline:
            self.LogText('\n', newLine=False, timestamp=False, quickinfo=quickinfo)
            
    #print scalar quantities of first level in dict:
    def PrintScalarQuantities(self, title, dataDict, separator=True, quickinfo=True, printToConsole=True):
        text = title+'\n'
        for key, value in dataDict.items():
            if IsScalar(value):
                text+=' - '+str(key)+' = '+str(value)+'\n'
                
        text+='\n'
        if separator: print('\n=======================================')
        self.LogText(text, separator=separator, quickinfo=quickinfo, printToConsole=printToConsole)

    def PrintAndLogCounters(self, separator=True, quickinfo=True, printToConsole=True):
        self.LogText(f'A total of:\n  {self.errorCount} errors and\n  {self.warningCount} warnings\nwere logged!', 
                separator=separator, quickinfo=quickinfo, printToConsole=printToConsole)

    def LogLLMresponse(self, llmModel, task, prompt, response, 
                       keyValues={}, logPrompt=True, thinkingMessage=None):
        task = str(task)
        response = str(response)
        
        self.LogText('LLM INFERENCE SUMMARY:\n', style=STYLE.bold, timestamp=True, separator=True)
        if task != '':
            self.LogText('LLM: task: '+task, style=STYLE.bold, timestamp=False, separator=False)

        if logPrompt:
            self.LogText('\nLLM: input prompt:', style=STYLE.bold, timestamp=False, separator=False)
            promptText = str(prompt)
            if isinstance(prompt, list): #only print content of user message for HF transformers
                for promptDict in prompt:
                    if (isinstance(promptDict, dict) 
                        and 'content' in promptDict 
                        and 'role' in promptDict
                        and promptDict['role'] == 'user'):
                        promptText = promptDict['content']
                
            self.LogText(promptText, style=STYLE.python, timestamp=False)

        self.LogText('LLM: response:', style=STYLE.bold, timestamp=False, separator=True)
        if thinkingMessage is not None and len(thinkingMessage.strip()) != 0:
            self.LogText('THINKING=\n'+thinkingMessage+'\n\nCONTENT=', 
                         style=STYLE.python, timestamp=False)
            
        self.LogText(response, style=STYLE.python, timestamp=False)

        backend = getattr(llmModel, 'backend', '').lower()
        # vllm reports exact prompt/generated token counts per request (from
        # RequestOutput.prompt_token_ids / outputs[].token_ids), so it belongs here too --
        # without it every vLLM run logged 'prompt tokens: "N/A"' and results.json recorded
        # no context statistics.
        tracksContextStats = backend in ('ollama', 'hftransformers', 'vllm')

        loggingDict = {
            "duration": round(llmModel.durationLocal, 2),
            "tokens generated": llmModel.numberOfTokensLocal,
            "thk tokens": llmModel.numberOfThinkingTokensLocal,
            "tokens per second": round(llmModel.tokensPerSecondLocal,3),
            "prompt tokens": llmModel.numberOfPromptTokensLocal if tracksContextStats else "N/A",
            "context size used": llmModel.contextSizeUsedLocal if tracksContextStats else "N/A",
            "context size max": llmModel.contextWindowMaxSize if tracksContextStats else "N/A",
        }
        loggingDict.update(keyValues)
        
        self.LogText('', timestamp=False, separator=False)
        self.LogDict(loggingDict, header='summary of '+task+':', inline=True, timestamp=False)
        self.LogText('', timestamp=False, separator=False)

#create class for styles
class STYLE :
    pass

STYLE.plain = 0
STYLE.bold = 1
STYLE.italics = 2
STYLE.python = 4
