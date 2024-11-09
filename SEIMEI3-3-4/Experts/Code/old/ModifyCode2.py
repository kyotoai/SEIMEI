import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ModifyCode2(Expert):

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):

        return {"ids":[], "keys":[]}


    async def inference(self, kwargs):

        query, chunk_id = kwargs["query"], kwargs["chunk_id"]

        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file: chunks = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)

        
        chunk = chunks[chunk_id]
        file_path = file_paths[chunk_id]
        prompt = f"""<s>[INST]You are an advanced language model tasked with analyzing a code chunk. You will be given a query, meta information of a code chunk, and the content of the chunk. Your task is to modify the code chunk as the query requires.


Given a query and a chunk of text, follow these steps:

1. **Analyze the Chunk**: Analyze the code and think deeply about the code step by step.
2. **Generate Output**: Provide your answer by following the answer format below;


Answer Format:
'''
### Deep Analysis:
(Your analysis here)

### Modified Code:
```
(Modified code here)
```
'''


Now, please analyze the following query and chunk:

### QUERY:
{query}

### CHUNK META INFO:
|-path: {file_path}
|-file summary: {f_summary[file_path]}

### CHUNK CONTENT:
```
{chunk}
```[/INST]"""

        llm = LLM(self)
        answer = await llm(prompt)  # to check if the chunk includes core of the simulation

        split_list = "### Modified Code:".split(answer)
        split_list2 = "```".split(split_list[-1])
        if len(split_list2) == 3:
            modified_code = split_list2[1]
        elif len(split_list2) == 2:
            if len(split_list2[0]) > len(split_list2[1]):
                modified_code = split_list2[0]
            else:
                modified_code = split_list2[1]
        else:
            raise Exception("something wrong with the answer")


        return {"modified_code":modified_code, "answer":answer}
