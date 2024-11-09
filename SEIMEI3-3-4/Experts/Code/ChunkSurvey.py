from SEIMEI import SEIMEI, LLM, Expert, Search
import json, os, re, asyncio

class ChunkSurvey(Expert):
    
    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True

    def get_keys(self):
        keys = []
        ids = []
        
        with open(f"/workspace/processed/{SEIMEI.database_name}/summary.json") as json_file: summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)

        for i in range(len(summary)):
            key = f"""This job will investigate the relation between user query and chunk information shown bellow.

CHUNK INFO
|-path:{file_paths[i]}
|-file summary:{f_summary[file_paths[i]]}
|-chunk summary:{summary[i]}"""
            
            keys.append(key)
            ids.append(i)
        
        return {"keys":keys, "ids":ids}
        #return {"keys":[], "ids":[]}


    async def inference(self, job_dict):

        query, id = job_dict["query"], job_dict["local_key_id"]

        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file: chunks = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file: file_paths = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file: f_summary = json.load(json_file)

        meta = f"""|-file path:{file_paths[id]}
|-file summary:{f_summary[file_paths[id]]}"""
        
        prompt = f"""<s>[INST]You are an excellent help assistant for investigating a large database. You will be provided with a chunk of text in the database. Based on the information from the database and question provided by the user, please take out relevant passage from the information and answer the provided question;

### QUESTION
{query}

### INFORMATION META
{meta}

### INFORMATION
```
{chunks[id]}
```

[/INST]"""

        llm = LLM(self)
        answer = await llm(prompt)

        return {"answer":answer}

