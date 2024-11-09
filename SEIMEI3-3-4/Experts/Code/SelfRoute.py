import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ChunkSurvey import ChunkSurvey
from FileSurvey2 import FileSurvey2


class SelfRoute(Expert):

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):

        keys = []
        ids = []

        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file: chunks = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/summary.json") as json_file: summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)

        for i in range(len(summary)):
            key = f"""This expert will judge if the chunk content below is enough to answer the question. If this expert judges the chunk content is enough, it will pass the chunk content directly to expert for answering the question. Otherwise; the whole file content will be passed to the answer expert.


### CHUNK META INFO
|-path: {file_paths[i]}
|-file summary: {f_summary[file_paths[i]]}
|-chunk summary: {summary[i]}"""
            
            keys.append(key)
            ids.append(i)
            
        return {"ids":ids, "keys":keys}


    async def inference(self, kwargs):

        query, id = kwargs["query"], kwargs["local_key_id"]
        #query, id, ids = kwargs["query"], kwargs["local_key_id"], kwargs["local_key_ids"]

        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file: chunks = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/summary.json") as json_file: summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)
        
        prompt = f"""<s>[INST]You are an advanced language model tasked with implementing the self-route approach. This approach involves first attempting to answer a query using Retrieval-Augmented Generation (RAG). If RAG cannot generate a satisfactory response, you will defer to Long-Context processing. Your goal is to balance the computational savings of RAG with the answer quality of long-context models.


Given a query and a chunk of text, follow these steps:

1. **Analyze the Chunk**: Evaluate whether the given chunk of text contains enough information to answer the query.
2. **Decision Making**: Decide whether to use RAG or defer to Long-Context processing based on the sufficiency of the chunk.
3. **Generate Output**: Provide your reasoning and decisions in the following JSON format:

```json
{{
  "thought": "(reason why the LLM made the following decision toward the given chunk)",
  "is the given chunk enough to answer the query": true or false,
  "is there likely to be any useful information in a file including the chunk": true or false
}}```


Now, please analyze the following query and chunk:


### QUERY:
{query}


### CHUNK META INFO:
|-path: {file_paths[id]}
|-file summary: {f_summary[file_paths[id]]}
|-chunk summary: {summary[id]}


### CHUNK CONTENT:
```
{chunks[id]}
```[/INST]"""

        llm = LLM(self)
        answer = await llm(prompt)  # to check if the chunk includes core of the simulation

        judge = 1 # default
        
        try:
            # Find the positions of the first '{' and the last '}'
            start_index = answer.find('{')
            end_index = answer.rfind('}')
            
            if start_index != -1 and end_index != -1 and start_index < end_index:
                # Extract the JSON part
                json_text = answer[start_index:end_index+1]
                
                json_output = json.loads(json_text)
                thought = json_output["thought"]
                judge1 = json_output["is the given chunk enough to answer the query"]
                judge2 = json_output["is there likely to be any useful information in a file including the chunk"]
                
                if judge1:
                    judge = 0  # given chunk is enought to answer the query
                elif judge2:
                    judge = 1  # given chunk is NOT enought to answer the query but the file including the chunk may include useful information
                else:
                    judge = 2  # the file will be irrelevant to the query
                
        except:
            print()
            print("--- json fail ----")
            print("answer: ", answer)
            judge = 1

        
        if judge == 0:
            chunk_survey = ChunkSurvey(self)
            output = await chunk_survey({"query":query, "local_key_id":id})
        elif judge == 1:
            file_survey = FileSurvey2(self)
            output = await file_survey({"query":query, "local_key_id":id})
        else:
            answer = f"""The following file is not likely to contain any useful information to answer the query.

**Query**: {query}

**File meta information**:
|-path: {file_paths[id]}
|-file summary: {f_summary[file_paths[id]]}"""

            output = {"answer":answer}

        SEIMEI.infs.append({"inf":output["answer"]})

        return output
