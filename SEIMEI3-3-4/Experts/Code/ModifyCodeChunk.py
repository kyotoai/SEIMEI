import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def extract_text_inside_backticks(text, arbitrary_text):
    # Define the pattern to match the text inside ``` that follows the arbitrary text
    pattern = re.compile(r'```{}\s*([\s\S]*?)\s*```'.format(re.escape(arbitrary_text)))

    # Search for the pattern in the text
    match = pattern.search(text)

    if match:
        return match.group(1).strip()
    else:
        return None


class ModifyCodeChunk(Expert):

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):
        return {"ids":[], "keys":[]}


    async def inference(self, kwargs):

        query, id = kwargs["query"], kwargs["chunk_id"]

        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file: chunks = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)

        chunk = chunks[id]

        prompt = f"""<s>[INST]### CODE SNIPPET:
```code
{chunk}
```


### FILE META INFO:
|-path: {file_paths[id]}
|-file summary: {f_summary[file_paths[id]]}


### QUERY:
```query
{query}
```


You are an advanced language model tasked with modifying the given code snippet as the given query demands. Please follow the instructions and output format below.


### Instructions:
1. **Analyze the Query, Code, and File Meta Information**: Carefully understand and analyze the provided query, code and file meta information in which the code is included, and think what to be modifed in the code.
2. **Generate Output**: Based on your judgment, return the output following the format below.


### Output Format:
Generate the output following the format below:

```modified code
(modified code here)
```


Let’s think step by step.[/INST]"""


        llm = LLM(self)
        answer = await llm(prompt)  # to check if the chunk includes core of the simulation

        modified_code = extract_text_inside_backticks(answer, "modified code")

        return modified_code
    