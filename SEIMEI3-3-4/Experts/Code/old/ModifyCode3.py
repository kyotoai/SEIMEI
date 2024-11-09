import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ChunkSurvey import ChunkSurvey

class ModifyCode3(Expert):

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):

        return {"ids":[], "keys":[]}


    async def inference(self, kwargs):

        query, chunk_id, next_action = kwargs["query"], kwargs["chunk_id"], kwargs["next_action"]

        search_result = SEIMEI.search([next_action], topk = 2, expert_restriction = [ChunkSurvey], prohibited_ids=[chunk_id])  # [[(ChunkSurvey, id1), (ChunkSurvey, id2)]]
        relevant_chunk_ids = [search_result[0][i][1] for i in range(len(search_result[0]))]

        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file: chunks = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)

        relevant_chunk_text = ""
        for id in relevant_chunk_ids:
            relevant_chunk_text += f"""```
{chunks[id]}
```

"""
                
        prompt = f"""### RELEVANT CHUNKS:
{relevant_chunk_text}

The code above maybe related to the following question. Use them if you need when answering the following question.


You are an advanced language model tasked with analyzing a code chunk. You will be given a query, meta information of a main code chunk and the content of the chunk. Your task is to modify the code chunk as the query requires.

Given a query and a chunk of text, follow these steps:

1. **Analyze the Chunk**: Analyze the main code and think deeply about the code step by step.
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

### MAIN CHUNK META INFO:
|-path: {file_paths[chunk_id]}
|-file summary: {f_summary[file_paths[chunk_id]]}

### CHUNK CONTENT:
```
{chunks[chunk_id]}
```[/INST]"""

        llm = LLM(self)
        answer = await llm(prompt)

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