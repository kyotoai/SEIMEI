import os, json, re, asyncio, traceback
from SEIMEI import SEIMEI, LLM, Expert, Search2

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MetaSurvey2 import MetaSurvey2
from QuickSummary import QuickSummary

f_dict = {}
root_paths = []
class F:
    def __init__(self, path):
        global f_dict, root_paths
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
        global f_dict, root_paths
        if self.parent != None:
            self.parent.children.append(self)
        else:
            if self.path not in root_paths:
                root_paths.append(self.path)

    def set_summary(self, summary):
        self.summary = summary


global_file_id = 0
global_file_list = []


class MetaModifyCode(Expert):

    description = "This expert analyzes some pieces of information which inference so far has gotten, and determine which files of code should be modified and what information to be used when modifying each file."
    
    def __init__(self, caller):
        super().__init__(caller)


    def get_file_path_text(self):
        global global_file_id, global_file_list
        
        with open(f"{SEIMEI.processed_path}/file_paths.json") as json_file:
            file_paths = json.load(json_file)

        all_file_paths = []
        for file_path in file_paths:
            if file_path not in all_file_paths:
                all_file_paths.append(file_path)


        for path in all_file_paths:
            F(path)

        for path in f_dict:
            f = f_dict[path]

            f.set_children()

        

        def recursive_get_file_path_text(path, depth):
            global global_file_id, global_file_list
            file_name = os.path.basename(path)
            
            if f_dict[path].children == []:
                text = depth*4*" " + str(global_file_id) + " " + file_name + "\n"
                global_file_id += 1
                global_file_list.append(path)
            else:
                text = depth*4*" " + file_name + "\n"
                
            for child in f_dict[path].children:
                text += recursive_get_file_path_text(child.path, depth+1)

            return text

        text = ""
        for root_path in root_paths:
            text += recursive_get_file_path_text(f_dict[root_path].path, 0)
        
        MetaModifyCode.file_paths = global_file_list
            
        return text



    def get_keys(self):
        MetaSurvey2.file_summary_dict = None
        MetaModifyCode.meta_text = self.get_file_path_text()

        key = f"""This expert plays an important role in analyzing the meta structure of database."""

        return {"ids":[0], "keys":[key]}


    async def inference(self, kwargs):

        if MetaSurvey2.file_summary_dict == None:
            quick_summary = QuickSummary(self)
            outputs = await quick_summary({"survey_paths":MetaModifyCode.file_paths})

            MetaSurvey2.file_summary_dict = {}

            for output in outputs:
                MetaSurvey2.file_summary_dict[output["file_path"]] = output["summary"]
            
        query, id = kwargs["query"], kwargs["local_key_id"]

        info_dicts = self.get_info(query, topk = 20)

        info_text = ""
        for i, info_dict in enumerate(info_dicts):
            info_text += f"""* information {i}:
```
{info_dict["info"]}
```

"""

        summary_text = ""
        for file_path in MetaSurvey2.file_summary_dict:
            summary_text += f"""* {file_path}:
```
{MetaSurvey2.file_summary_dict[file_path]}
```

"""

        prompt = f"""### INFORMATION:
{info_text}

### FILE SUMMARY:
{summary_text}

### META STRUCTURE:
```meta
{MetaModifyCode.meta_text}
```


### QUERY:
```query
{query}
```


You are an advanced language model tasked with identifying the file which should be modified to answer the given query and designate which pieces of information to be used to modify the file. You are provided with some pieces of information, file summary, meta structure of a database and query above. The information and files have its index at the beginning of them. Please follow the instructions and output the format below.


### Instructions:
1. **Analyze the Information, File Summary, Meta Structure and Query**: Carefully read the provided text, and figure out how files works in the system and relevant to the query.
2. **Decision**: Based on your analysis, determine which file should be modified to answer the query and which pieces of information are needed to modify.
3. **Generate Output**: Based on your decision, return the indices of files and pieces of information, and instruction of how to modify the file following the output format below.


### Output Format:
'''
(Your careful analysis and decision here)

```json
[
    {{
        "file id": (id at the beginning of a file to modify),
        "instruction": "(instruction of how to modify the file)",
        "info ids": [ id1 (id on top of each info necessary to the modification), ... ]
    }},
    ...
]
```
'''


Let’s think step by step."""

        llm = LLM(self)
        output = await llm(prompt)

        json_text = SEIMEI.extract_text_inside_backticks(output, "json")

        try:
            json_data = json.loads(json_text)

            experts = []
            for i in range(len(json_data)):
                file_id = json_data[i]["file id"]
                instruction = json_data[i]["instruction"]
                info_ids = json_data[i]["info ids"]

            SEIMEI.final_answer = output
            SEIMEI.answer_end = True

        except Exception as e:
            traceback.print_exc()





