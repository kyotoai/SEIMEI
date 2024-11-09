import os, json, re, asyncio
from SEIMEI import SEIMEI, LLM, Expert

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from FileSurvey import FileSurvey


f_dict = {}
class F:
    def __init__(self, path):
        if path not in f_dict:
            self.children = []
            self.path = path
            dirname = os.path.dirname(path)
            if dirname != "":
                if dirname in f_dict: self.parent = f_dict[dirname]
                else:
                    self.parent = F(dirname)
            else:
                self.parent = None
    
            f_dict[self.path] = self
    
    def set_children(self):
        if self.parent != None:
            self.parent.children.append(self)

    def set_summary(self, summary):
        self.summary = summary



class MetaSurvey(Expert):
    
    def __init__(self, caller):
        super().__init__(caller)
        
        #self.file_survey = FileSurvey(self.seimei)
        #self.file_surveys = [FileSurvey(self.seimei) for _ in range(10)]


    def get_keys(self):
        with open(f"/workspace/processed/{SEIMEI.database_name}/f_summary.json") as json_file:
            f_summary = json.load(json_file)

        for path in f_summary:
            F(path)

        if len(f_summary) != len(f_dict):
            raise Exception("error")
            
        for path in f_summary:
            summary = f_summary[path]
            f = f_dict[path]

            f.set_summary(summary)
            f.set_children()

        id = 0
        ids = []
        keys = []
        MetaSurvey.paths = {}

        for path in f_summary:
            if f_dict[path].children != []:
                MetaSurvey.paths[id] = path

                key = f"""This job will figure out what folder or file is related to user question. In this job, files in directory `{path}` will be checked.

Meta Infomation of `{path}`:
{f_summary[path]}"""

                keys.append(key)
                ids.append(id)

                id += 1
            
        return {"ids":ids, "keys":keys}
        #return {"ids":[], "keys":[]}
    

    async def inference(self, kwargs):
        
        query, id = kwargs["query"], kwargs["local_key_id"]
        path = MetaSurvey.paths[id]

        childen_info = ""
        for i in range(len(f_dict[path].children)):
            childen_info += f"""{i} `{f_dict[path].children[i].path}`: {f_dict[path].children[i].summary}
"""

        prompt = f"""<s>[INST]Parent Folder `{path}`: {f_dict[path].summary}

Files:
'''
{childen_info}
'''

Question:{query}

Which files are likely to include the information relevant to the question? Answer by the json format below;
{{
    "reasons":(reasons why the following files are relevant to the question),
    "question_relevant_file_path":[(list of file paths relevant to the question)]
}}[/INST]"""

        print("MetaSurvey prompt: ", SEIMEI.get_num_tokens(prompt))

        prompt = SEIMEI.cut_text(prompt)
        
        llm = LLM(self)
        output = await llm(prompt)

        print("MetaSurvey output: ", SEIMEI.get_num_tokens(output))

        try:

            # Find the positions of the first '{' and the last '}'
            start_index = output.find('{')
            end_index = output.rfind('}')
            
            if start_index != -1 and end_index != -1 and start_index < end_index:
                # Extract the JSON part
                json_text = output[start_index:end_index+1]
                    
                #pattern = r"\{([^}]*)\}"
                #matches = re.findall(pattern, answer)
                #answer_text = "{" + matches[0] + "}"
            
                json_output = json.loads(json_text)
                #reasons = json_output["reasons"]
                survey_paths = json_output["question_relevant_file_path"]

        except:
            print("json fail")
            #reasons = answer
            survey_paths = []

        print("MetaSurvey survey_paths: ", survey_paths)

        with open(f"/workspace/processed/{SEIMEI.database_name}/file_paths.json") as json_file:
            file_paths = json.load(json_file)

        # This is because sometimes survey_path is like "transformers/docs/*.md"
        def wildcard_match(pattern, string):
            regex_pattern = pattern.replace("*", ".*")
            return re.fullmatch(regex_pattern, string) is not None

        survey_paths_ = []
        for survey_path in survey_paths:
            for file_path in file_paths:
                if wildcard_match(survey_path, file_path):
                    if not file_path in survey_paths_:
                        survey_paths_.append(file_path)
                        
        return {"survey_paths": survey_paths_}

    
    

