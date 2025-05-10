import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ChunkSurvey import ChunkSurvey
from FileSurvey2 import FileSurvey2


class SelfRoute(Expert):

    description = "This expert judges if a given chunk of text is enough to answer user's query. If this expert judges it's enough, this expert will pass the chunk of text to ChunkSurvey; otherwise, it will pass to FileSurvey2, where all text in a file is investigated."

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):

        keys = []
        ids = []

        with open(f"{SEIMEI.processed_path}/file_paths.json") as json_file: file_paths = json.load(json_file)

        for i in range(len(file_paths)):
            key = f"""This expert judges if a given chunk of text is enough to answer user's query. If this expert judges it's enough, this expert will pass the chunk of text to ChunkSurvey; otherwise, it will pass to FileSurvey2, where all text in a file is investigated.

### CHUNK META INFO
|-path: {file_paths[i]}"""
            
            keys.append(key)
            ids.append(i)
            
        return {"ids":ids, "keys":keys}


    async def inference(self, kwargs):

        query, id = kwargs["query"], kwargs["local_key_id"]
        #query, id, ids = kwargs["query"], kwargs["local_key_id"], kwargs["local_key_ids"]

        with open(f"{SEIMEI.processed_path}/chunks.json") as json_file: chunks = json.load(json_file)
        with open(f"{SEIMEI.processed_path}/file_paths.json") as json_file: file_paths = json.load(json_file)
        
        prompt = f"""### CHUNK CONTENT:
```
{chunks[id]}
```

### CHUNK META INFO:
|-path: {file_paths[id]}

### QUERY:
{query}


You are an advanced language model tasked with judging if a text chunk includes enough information to answer a query. You are given a text chunk from an article, and a query. The text chunk is retrieved by an external retriever. Please follow the instructions and output format below.


### Instructions: 
1. **Analyze the Chunk**: Evaluate whether the given chunk of text contains enough information to answer the query.
2. **Decision Making**: Decide next action from the 3 options. 1. to give only the chunk to answering LLM, 2. to extract more information from database, 3. to discard the text chunk due to its irrelevance to the query.
3. **Generate Output**: Provide your reasoning and decisions in the following JSON format:


### Output Format:

‘’’
(Your careful analysis here)

```decision
(Option number)
```
‘’’


Let’s think step by step."""

        llm = LLM(self)
        answer = await llm(prompt)  # to check if the chunk includes core of the simulation

        decision_id_text = SEIMEI.extract_text_inside_backticks(answer, "decision")

        try:
            decision_id = int(decision_id_text)

        except:
            print()
            print("--- Fail ----")
            print("answer: ", answer)
            decision_id = 3

        
        if decision_id == 1:
            chunk_survey = ChunkSurvey(self)
            output = await chunk_survey({"query":query, "local_key_id":id})
        elif decision_id == 2:
            file_survey = FileSurvey2(self)
            output = await file_survey({"query":query, "local_key_id":id})
        else:
            answer = f"""The following file is not likely to contain any useful information to answer the query.

**Query**: {query}

**File meta information**:
|-path: {file_paths[id]}"""

            output = {"answer":answer}

        self.set_info(info = output["answer"], query = query)

        return output
