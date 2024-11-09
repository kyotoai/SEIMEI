import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ModifyCodeChunk import ModifyCodeChunk
from SummarizeCode import SummarizeCode

def extract_text_inside_backticks(text, arbitrary_text):
    # Define the pattern to match the text inside ``` that follows the arbitrary text
    pattern = re.compile(r'```{}\s*([\s\S]*?)\s*```'.format(re.escape(arbitrary_text)))

    # Search for the pattern in the text
    match = pattern.search(text)

    if match:
        return match.group(1).strip()
    else:
        return None


class ModifyCodeFile(Expert):

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):

        keys = []
        ids = []
        
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)

        ModifyCodeFile.file_paths = []

        for i, file_path in enumerate(f_summary):
            key = f"""This expert will try to modify code in a file whose meta information is written below.

### FILE META INFO
|-path: {file_path}
|-file summary: {f_summary[file_path]}"""
            
            keys.append(key)
            ids.append(i)
            ModifyCodeFile.file_paths.append(file_path)

        return {"ids":ids, "keys":keys}


    async def inference(self, kwargs):

        query, id = kwargs["query"], kwargs["local_key_id"]
        survey_file_path = ModifyCodeFile.file_paths[id]

        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file: chunks = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)

        experts = []
        chunk_ids = []
        for i, file_path in enumerate(file_paths):
            if file_path == survey_file_path:
                summarize_code = SummarizeCode(self)
                experts.append(summarize_code({"chunk_id":i}))
                chunk_ids.append(i)

        code_summaries = await asyncio.gather(*experts)
        
        code_summaries_text = ""
        for i, code_summary in enumerate(code_summaries):
            code_summaries_text += f"""```code summary {i}
{code_summary}
```\n\n"""

        prompt = f"""<s>[INST]### CODE SNIPPETS:
{code_summaries_text}


### FILE META INFO:
|-path: {survey_file_path}
|-file summary: {f_summary[survey_file_path]}


### QUERY:
```query
{query}
```


You are an advanced language model tasked with identifying the part of the code that should be modified based on a given query. You are provided with several code summaries above, which are summarized from original code snippets. Please follow the instructions and output format below.


### Instructions:
1. **Analyze the Query and Code**: Carefully read and understand the provided query and code summaries.
2. **Judge**: Based on your analysis, determine which code summary should be modified to address the query.
3. **Generate Output**: Based on your judgment, return the output following the format below.


### Output Format:
Generate the output following the format below:

```list of indices
[ number1, number2, … ]
```


Let’s think step by step.[/INST]"""


        llm = LLM(self)
        answer = await llm(prompt)  # to check if the chunk includes core of the simulation


        number_list_text = extract_text_inside_backticks(answer, "list of indices")
        number_list = json.loads(number_list_text)

        experts = []
        chunk_ids2 = []
        for number in number_list:
            id = chunk_ids[number]
            chunk_ids2.append(id)
            modify_code = ModifyCodeChunk(self)
            experts.append(modify_code({"query":query, "chunk_id":id}))

        modified_codes = await asyncio.gather(*experts)

        for i, modified_code in enumerate(modified_codes):

            inf = f"""This is the modified code for a code of chunk. The meta chunk information and modified code are weitten below;

### FILE META INFO:
|-path: {survey_file_path}
|-file summary: {f_summary[survey_file_path]}
|-chunk id: {chunk_ids2[i]}

### MODIFIED CODE
{modified_code}
"""

            SEIMEI.infs.append({"inf":inf})