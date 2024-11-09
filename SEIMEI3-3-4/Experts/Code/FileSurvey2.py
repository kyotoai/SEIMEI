import os, json, re, asyncio
import copy
from SEIMEI import SEIMEI, LLM, Expert


class FileSurvey2(Expert):

    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True


    def get_keys(self):

        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file:
            file_paths = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file:
            f_summary = json.load(json_file)

        id = 0
        ids = []
        keys = []
        self.file_paths = []
        for file_path in file_paths:
            if file_path not in self.file_paths:
                key = f"""This job will investigate a file `{file_path}`.

Meta Infomation of `{file_path}`:
{f_summary[file_path]}"""
                keys.append(key)
                self.file_paths.append(file_path)
                ids.append(id)
                id += 1


        #return {"ids":ids, "keys":keys}
        return {"ids":[], "keys":[]}


    async def inference(self, kwargs):

        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file:
            file_paths = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file:
            chunks = json.load(json_file)
        
        query, id = kwargs["query"], kwargs["local_key_id"]
        survey_path = file_paths[id]

        file_content = ""
        for i in range(len(file_paths)):
            if file_paths[i] == survey_path:
                file_content += chunks[i]

        cut_file_content = SEIMEI.cut_text(file_content, num_token = 5000, cut_back = False)
        

        prompt = f"""<s>[INST]You are a helpful assistant for analyzing a file and extracting useful information from it. You'll be given a question and some information from a file.

**File content**: 
```
{cut_file_content}
```

**File path**: {survey_path}


**Query**: {query}


Please answer the query as much in detail as possible using the given file content.[/INST]"""

        llm = LLM(self)
        answer = await llm(prompt)

        
        return {"answer":answer}

