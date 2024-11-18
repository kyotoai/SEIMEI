import os, json, re, asyncio, traceback
from SEIMEI import SEIMEI, LLM, Expert

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ModifyCodeFile import ModifyCodeFile

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


class MetaSurvey(Expert):
    
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
            #summary = f_summary[path]
            f = f_dict[path]

            #f.set_summary(summary)
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
        
        MetaSurvey.file_paths = global_file_list
            
        return text



    def get_keys(self):
        MetaSurvey.meta_text = self.get_file_path_text()
        MetaSurvey.info_dict = {}  # {"query1": [{"inf": }, ...], ...}

        key = f"""This expert plays an important role in analyzing the meta structure of database."""

        return {"ids":[0], "keys":[key]}


    async def inference(self, kwargs):

        query, id = kwargs["query"], kwargs["local_key_id"]

        #if query in MetaSurvey.info_dict:
        #    infs = MetaSurvey.info_dict[query]
        #else:
        #    infs = SEIMEI.search_inf(query, 3)

        info_dicts = self.get_info(query, topk = 5)

        info_text = ""
        for i, info_dict in enumerate(info_dicts):
            info_text += f"""```information {i}
{info_dict["info"]}
```

"""


        prompt = f"""<s>[INST]### INFORMATION:
{info_text}

### META STRUCTURE:
```meta
{MetaSurvey.meta_text}
```


### QUERY:
```query
{query}
```


You are an advanced language model tasked with identifying the file which should be investigated to answer the given query. You are provided with information, meta structure of a database and query above. Each file has its index at the beginning of the file name. Please follow the instructions and output format below.


### Instructions:
1. **Analyze the Information, Meta Structure and Query**: Carefully read the provided information, meta structure and query, and guess how each file works in the system.
2. **Decision**: Based on the provided information, names of file path, and common sense, determine which file should be investigated to answer the query. To make the decision, you don't necessarily use the given information.
3. **Generate Output**: Based on your decision, return the indices written before the file name to be investigated following the output format below.


### Output Format:

```list of indices
[ number1, number2, … ]
```


Let’s think step by step.[/INST]"""

        llm = LLM(self)
        output = await llm(prompt)

        indices_text = SEIMEI.extract_text_inside_backticks(output, "list of indices")

        try:
            indices = json.loads(indices_text)

            experts = []
            for id in indices:
                modify_code_file = ModifyCodeFile(self)
                experts.append(modify_code_file({"query":query, "survey_path":MetaSurvey.file_paths[id]}))

            await asyncio.gather(*experts)

        except Exception as e:
            traceback.print_exc()





