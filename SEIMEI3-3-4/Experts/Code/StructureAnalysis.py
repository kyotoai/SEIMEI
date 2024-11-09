import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MetaSurvey import MetaSurvey


class StructureAnalysis(Expert):

    def __init__(self, caller):
        super().__init__(caller)
        self.llm1 = LLM(self)
        self.meta_survey = MetaSurvey(self)


    def get_keys(self):
        meta_dict = self.meta_survey.get_keys()

        new_keys = []
        for i, key in enumerate(meta_dict["keys"]):
            new_key = key + "\n\nThis job will also analyze the structure of code, and tell you where in the code algorithm starts and the calculation goes through step by step."
            new_keys.append(key)
            
        return {"ids":meta_dict["ids"], "keys":new_keys}


    async def inference(self, kwargs):
        #query = "What's the core of the entire database? By core, I mean a part of database which controls entire files. For example, time step loop normally plays an important role in some simulation code, so it can be said a core."

        query = "To figure out how the code is run, which file is likely to include information about the order of running code?"

        # ここでsearchJobをmetaに限ってやる必要がある
        # output = self.search_job({"query":query, "job_restriction":[MetaSurvey], "auto_job_set":False}) # {"kwargs_list":[kwargs1, ...], "job_class_list":[job_class1, ...]}
        kwargs["query"] = query
        result = await self.meta_survey(kwargs)
        survey_paths = result["survey_paths"]

        #print("StructureAnalysis survey_paths: ", survey_paths)

        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file:
            f_summary = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file:
            file_paths = json.load(json_file)
        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file:
            chunks = json.load(json_file)

        prompts = []
        chunk_ids = []
        once_surveyed = []
        for survey_path in survey_paths:
            for i, file_path in enumerate(file_paths):
                if file_path == survey_path and not survey_path in once_surveyed:
                    once_surveyed.append(survey_path)
                    prompt = f"""<s>[INST]You are an excellent assistant and are adept at investigating a database. You will be provided with a chunk of data in the database with its meta information.

### File Path
'{survey_path}'

### File Meta Information
{f_summary[survey_path]}

### Chunk
{chunks[i]}

You should check if the chunk includes some core function in the entire database. For example, time step loop normally plays an important role in some simulation code, so it can be said a core. Please answer that following the json format bellow;
{{
    "thought":(reason of the judge. Ex. what the chunk is about.),
    "judge":true or false. (true if the chunk includes some core function in the entire database.)
}}[/INST]"""

                    prompts.append(prompt)
                    chunk_ids.append(i)

        answers = await self.llm1(prompts)  # to check if the chunk includes core of the simulation

        
        output = {"chunk_ids":[]}
        for j, answer in enumerate(answers):
            #print()
            #print(j)
            #print("prompt: ")
            #print(prompts[j])
            #print("answer: ")
            #print(answer)
            
            try:

                # Find the positions of the first '{' and the last '}'
                start_index = answer.find('{')
                end_index = answer.rfind('}')
                
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    # Extract the JSON part
                    json_text = answer[start_index:end_index+1]
                        
                    #pattern = r"\{([^}]*)\}"
                    #matches = re.findall(pattern, answer)
                    #answer_text = "{" + matches[0] + "}"
                    
                    # json_output = re.search(r'{.*}', all_outputs[0]["generated_text"]).group(0)
                    # json_output = re.findall(r'\{.*?\}', all_outputs[0]["generated_text"], re.DOTALL)[0]
                    json_output = json.loads(json_text)
                    thought = json_output["thought"]
                    judge = json_output["judge"]
                    if judge:
                        output["chunk_ids"].append(chunk_ids[j])
                
            except:
                print()
                print("--- json fail ----")
                print("answer: ", answer)

        return output
