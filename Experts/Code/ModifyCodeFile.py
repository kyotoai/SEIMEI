import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Experts.Code.ModifyCodeChunk import ModifyCodeChunk
from Experts.Code.SummarizeCode import SummarizeCode
from Experts.Code.PreprocessFile import PreprocessFile


class ModifyCodeFile(Expert):

    description = "This expert modifies whole code included in a file."

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):

        keys = []
        ids = []

        '''
        #with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"{SEIMEI.processed_path}/file_paths.json") as json_file: file_paths = json.load(json_file)

        ModifyCodeFile.file_paths = []
        id = 0
        for file_path in file_paths:
            if file_path in ModifyCodeFile.file_paths:
                key = f"""This expert trys to modify code in a file whose meta information is written below.

### FILE META INFO
|-path: {file_path}"""

                keys.append(key)
                ids.append(id)
                id += 1
                ModifyCodeFile.file_paths.append(file_path)

        '''
        
        return {"ids":ids, "keys":keys}


    async def inference(self, kwargs):

        if "survey_path" in kwargs:
            query = kwargs["query"]
            survey_file_path = kwargs["survey_path"]
        else:
            query, id = kwargs["query"], kwargs["local_key_id"]
            survey_file_path = ModifyCodeFile.file_paths[id]

        #with open(f"{SEIMEI.processed_path}/chunks.json") as json_file: chunks = json.load(json_file)
        #with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        #with open(f"{SEIMEI.processed_path}/file_paths.json") as json_file: file_paths = json.load(json_file)

        preprocess = PreprocessFile(self)
        chunks = await preprocess(kwargs)

        experts = []
        for i, chunk in enumerate(chunks):
            summarize_code = SummarizeCode(self)
            experts.append(summarize_code({"chunk":chunk}))

        code_summaries = await asyncio.gather(*experts)
        
        code_summaries_text = ""
        for i, code_summary in enumerate(code_summaries):
            code_summaries_text += f"""```code summary {i}
{code_summary}
```\n\n"""

        prompt = f"""### CODE SUMMARIES:
{code_summaries_text}


### FILE META INFO:
|-path: {survey_file_path}


### QUERY:
```query
{query}
```


You are an advanced language model tasked with identifying the part of the code that should be modified based on a given query. You are provided with some code summaries above, which are summarized from original code snippets. Please follow the instructions and output format below.


### Instructions:
1. **Analyze the Query and Code Summaries**: Carefully read and understand the provided query and code summaries.
2. **Judge**: Based on your analysis, determine which code summary should be modified to address the query.
3. **Generate Output**: Based on your judgment, return indices at the beginning of each code summary following the format below.


### Output Format:
Generate the output following the format below:

‘’’
(Your careful analysis here)

```list of indices
[ id1, id2, ... ]
```
‘’’


Let’s think step by step following the instructions above."""


        llm = LLM(self)
        answer = await llm(prompt)  # to check if the chunk includes core of the simulation

        print()
        print("MetaCodeFile prompt : ", prompt)
        print()
        print("MetaCodeFile answer : ", answer)


        number_list_text = SEIMEI.extract_text_inside_backticks(answer, "list of indices")
        number_list = json.loads(number_list_text)

        print("0")

        experts = []
        #chunk_ids2 = []
        for number in number_list:
            try:
                #id = chunk_ids[number]
                #chunk_ids2.append(id)
                chunk = chunks[number]
                modify_code = ModifyCodeChunk(self)
                experts.append(modify_code({"query":query, "chunk":chunk, "file_path":survey_file_path}))
                print("2")
            except Exception as e:
                traceback.print_exc()

        output_dicts = await asyncio.gather(*experts)

        experts = []
        for i, output_dict in enumerate(output_dicts):
            if output_dict["success"]:
                if "modified_code" in output_dict:
                    info = f"""This is the modified code for a code of chunk. The meta chunk information and modified code are written below;

### CHUNK META INFO:
|-path: {survey_file_path}

### MODIFIED CODE
{output_dict["modified_code"]}
"""
                    self.set_info(info, query)

                '''
                elif "next_action" in output_dict:
                    search = Search(self, num_expert = 1)
                    experts.append(search({"queries":[output_dict["next_action"]]}))
                '''

        await asyncio.gather(*experts)

