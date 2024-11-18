import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio


class ModifyCodeChunk(Expert):

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):
        keys = []
        ids = []

        #with open(f"/workspace/processed/{SEIMEI.database_name}/summary.json") as json_file: summary = json.load(json_file)
        #with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"{SEIMEI.processed_path}/file_paths.json") as json_file: file_paths = json.load(json_file)

        for i, file_path in enumerate(file_paths):
            key = f"""This expert will try to modify code chunk in a file. The information of chunk is written below;

### CHUNK META INFO
|-path: {file_path}"""

            ids.append(i)
            keys.append(key)

        
        return {"ids":ids, "keys":keys}


    async def inference(self, kwargs):

        query, id = kwargs["query"], kwargs["chunk_id"]

        with open(f"{SEIMEI.processed_path}/chunks.json") as json_file: chunks = json.load(json_file)
        #with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"{SEIMEI.processed_path}/file_paths.json") as json_file: file_paths = json.load(json_file)

        chunk = chunks[id]
        file_path = file_paths[id]

        prompt = f"""<s>[INST]### INFO:
```info
{self.get_info()}
```


### CODE:
```code
{chunk}
```


### FILE META INFO:
```meta
|-path: {file_path}
```


### QUERY: 
```query
{query}
``` 


You are an advanced language model tasked with modifying the given code snippet as the given query demands. If there is not enough information to answer the question you should instead ask for more information. Please follow the instructions and output format below. 


### Instructions: 
1. **Analyze the Code, Information and Query**: Carefully understand and analyze the provided info, code and query, and think what to be modified in the code very carefully. 
2. **Judgement**: Based on your analysis, judge if the code should be modified or not. If there is not enough information to modify the code, think what information is needed to answer the query. 
3. **Generate Output**: Based on your analysis and judgement, return the output following the format below. 


### Output Format: 
Generate the output following the format below:
'''
(Very Careful Analyzation Here)

```judge
true or false (true if modification is needed; otherwise false)
```

```code
(modified code here. if there is not enough information, leave here blank) 
```

```next action
(if there is not enough information to modify the code)
```
'''

Let’s think step by step.[/INST]"""


        llm = LLM(self)
        answer = await llm(prompt)  # to check if the chunk includes core of the simulation

        print()
        print()
        print("ModifyCodeChunk")
        print("answer")
        print(answer)

        judge_text = SEIMEI.extract_text_inside_backticks(answer, "judge")
        modified_code = SEIMEI.extract_text_inside_backticks(answer, "code")
        next_action = SEIMEI.extract_text_inside_backticks(answer, "next action")

        
        print()
        print("modified_code")
        print(modified_code)
        print()
        print("next_action")
        print(next_action)
        print()
        print("judge_text")
        print(judge_text)

        if "true" in judge_text or "True" in judge_text:
            judge = True
        elif "false" in judge_text or "False" in judge_text:
            judge = False
        elif judge == None:
            judge = False
        else:
            judge = False
            print("something is wrong")

        if judge:
            if modified_code:
                return {"modified_code":modified_code, "success":True}
            else:
                return {"modified_code":modified_code, "success":False}
        else:
            if next_action:
                return {"next_action":next_action, "success":True}
            else:
                return {"next_action":next_action, "success":False}

