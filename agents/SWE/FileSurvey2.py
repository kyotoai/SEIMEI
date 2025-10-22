import os, json, re, asyncio
import copy
from SEIMEI import SEIMEI, LLM, Expert


class FileSurvey2(Expert):

    description = "This expert investigates a file content and get some useful information to a given query."

    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True


    def get_keys(self):

        return {"ids":[], "keys":[]}


    async def inference(self, kwargs):

        with open(f"{SEIMEI.processed_path}/file_paths.json") as json_file:
            file_paths = json.load(json_file)
        with open(f"{SEIMEI.processed_path}/chunks.json") as json_file:
            chunks = json.load(json_file)

        if "survey_path" in kwargs:
            query, survey_path = kwargs["query"], kwargs["survey_path"]
        else:
            query, id = kwargs["query"], kwargs["local_key_id"]
            survey_path = file_paths[id]

        file_content = ""
        for i in range(len(file_paths)):
            if file_paths[i] == survey_path:
                file_content += chunks[i]

        cut_file_content = SEIMEI.cut_text(file_content, num_token = 30000, cut_back = False)
        

        prompt = f"""You are a helpful assistant for analyzing a file and extracting useful information from it. You'll be given a question and some information from a file.

**File content**: 
```
{cut_file_content}
```

**File path**: {survey_path}


**Query**: {query}


Please answer the query as much in detail as possible using the given file content."""

        llm = LLM(self)
        answer = await llm(prompt)

        #print("FileSurvey2 answer: ", answer)

        self.set_info(info = answer, query = query)

        
        return {"answer":answer}

