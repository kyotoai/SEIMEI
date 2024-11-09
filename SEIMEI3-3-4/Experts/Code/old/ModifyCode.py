import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ModifyCode2 import ModifyCode2
from ModifyCode3 import ModifyCode3


class ModifyCode(Expert):

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):

        keys = []
        ids = []
        
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)

        ModifyCode.file_paths = []

        for i, file_path in enumerate(f_summary):
            key = f"""This expert will try to modify the code of the file whose meta information is written below.

### FILE META INFO
|-path: {file_path}
|-file summary: {f_summary[file_path]}"""
            
            keys.append(key)
            ids.append(i)
            ModifyCode.file_paths.append(file_path)


        return {"ids":ids, "keys":keys}


    async def inference(self, kwargs):

        query, id = kwargs["query"], kwargs["local_key_id"]
        survey_file_path = ModifyCode.file_paths[id]

        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file: chunks = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/summary.json") as json_file: summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)

        prompts = []
        chunk_ids = []
        for i, file_path in enumerate(file_paths):
            if file_path == survey_file_path:
                chunk = chunks[i]
                chunk_ids.append(i)
                
                prompt = f"""<s>[INST]You are an advanced language model tasked with analyzing a code chunk. You will be given a query, meta information of a code chunk, and the content of the chunk. Your task is to check if the code chunk should be modified or not and return the modified code if yes. But since the code modification task requires highly careful analysis, you should analyze another chunk of the code in the file and combine the information and change it. So, if you find the necessity, tell me what code to analyze next or what action we should take for more detailed analysis.

Given a query and a chunk of text, follow these steps:

1. **Analyze the Chunk**: Evaluate whether the given chunk of text contains enough information to answer the query.
2. **Decision Making**: Decide whether to modify the chunk directly or analyze another code for making more accurate code modification.
3. **Generate Output**: Provide your reasoning and decisions in the following JSON format:

```json
{{
  "analysis": "(careful and detailed analysis of the given code chunk.)",
  "modification": true or false (return true if the given code chunk needs to be modified.),
  "start modification": true or false (return true if the given code is self consistent enough to be modified without additional information.),
  "next action": "(what you should do next under the decision you made. If you put false in the decision above, return what to investigate next.)"
}}```


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
                prompts.append(prompt)

        llm = LLM(self)
        answers = await llm(prompts)  # to check if the chunk includes core of the simulation


        judges1 = []
        judges2 = []
        next_actions = []
        inferences = []
        for i, answer in enumerate(answers):
        
            try:
                # Find the positions of the first '{' and the last '}'
                start_index = answer.find('{')
                end_index = answer.rfind('}')
                
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    # Extract the JSON part
                    json_text = answer[start_index:end_index+1]
                    
                    json_output = json.loads(json_text)
                    analysis = json_output["analysis"]
                    judge1 = json_output["modification"]
                    judge2 = json_output["start modification"]
                    next_action = json_output["next action"]
                    
            except:
                print()
                print("--- json fail ----")
                print("answer: ", answer)
                judge1 = False
                judge2 = False
                next_action = ""

            if judge1:
                if judge2:
                    modify_code2 = ModifyCode2(self)
                    inferences.append(modify_code2({"query":query, "chunk_id":chunk_ids[i]}))
                else:
                    modify_code3 = ModifyCode3(self)
                    inferences.append(modify_code3({"query":query, "chunk_id":chunk_ids[i], "next_action":next_action}))

        outputs = await asyncio.gather(*inferences)


        for output in outputs:
            SEIMEI.infs.append({"inf":output["answer"]})