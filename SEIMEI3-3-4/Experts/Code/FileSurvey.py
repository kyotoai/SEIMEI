import os, json, re, asyncio
import copy
from SEIMEI import SEIMEI, LLM, Expert

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ChunkSurvey import ChunkSurvey


class FileSurvey(Expert):
    
    def __init__(self, caller):
        super().__init__(caller)
        

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
                
        
        return {"ids":ids, "keys":keys}
    

    async def inference(self, kwargs):
        
        query = kwargs["query"]
        if "file_path" in kwargs:
            survey_path = kwargs["file_path"]
        elif "local_key_id" in kwargs:
            survey_path = self.file_paths[kwargs["local_key_id"]]
            print("FileSurvey survey_path: ", survey_path)
        else:
            raise Exception("kwargs should include file_path or local_key_id")
        
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file:
            file_paths = json.load(json_file)

        answers = []
        num_chunk_survey = 0
        kwargs_list = []
        for i in range(len(file_paths)):
            if file_paths[i] == survey_path:
                num_chunk_survey += 1
                kwargs = copy.deepcopy(kwargs)
                kwargs["local_key_id"] = i
                kwargs_list.append(kwargs)

        inferences = []
        for i in range(len(kwargs_list)):
            chunk_survey = ChunkSurvey(self)
            inferences.append(chunk_survey(kwargs_list[i]))

        answers = await asyncio.gather(*inferences)  # [{..}, ...]
        #answers = await chunk_survey(kwargs_list)  # [{..}, ...]

        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file:
            f_summary = json.load(json_file)

        prompt = f"""<s>[INST]You are a helpful assistant for analyzing a file and extracting useful information from it. You'll be given a question and some information from a file.

QUESTION: {query}

FILE META INFORMATION: {f_summary[survey_path]}
"""
        
        for i in range(len(answers)):
            prompt += f"""
INFORMATION OF CHUNK {i}:
{answers[i]["answer"]}
"""
            
        prompt += f"""
Please answer the question as much in detail as possible by using all the useful intel in the file.[/INST]"""

        llm = LLM(self)
        answer = await llm(prompt)

        return {"answer":answer}

