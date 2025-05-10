import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio


class QuickSummary(Expert):

    description = "This expert makes quick summaries of some files."

    def __init__(self, caller, num_token_head = 2000, batch_size = 5):
        super().__init__(caller)
        self.log_off__ = True
        self.num_token_head = num_token_head
        self.batch_size = batch_size


    def get_keys(self):

        return {"ids":[], "keys":[]}


    async def inference(self, kwargs):

        survey_paths___ = kwargs["survey_paths"]

        survey_paths__ = []
        for key in survey_paths___:
            survey_paths__.append(survey_paths___[key])

        # with open(f"{SEIMEI.processed_path}/chunks.json") as json_file: chunks = json.load(json_file)
        # with open(f"{SEIMEI.processed_path}/file_paths.json") as json_file: file_paths = json.load(json_file)

        prompts = []
        survey_paths_list = []
        for j in range(len(survey_paths__)//self.batch_size + 1):
            survey_paths_ = survey_paths__[self.batch_size * j : self.batch_size * (j+1)]
            file_heads_text = ""
            survey_paths = {}
            for file_id, survey_path in enumerate(survey_paths_):
                try:
                    with open(survey_path) as f:
                        content = f.read()
                    head_text = SEIMEI.cut_text(content, num_token = self.num_token_head, cut_back = False)
                    survey_paths[file_id] = survey_path
                    file_heads_text += f"""file id: {file_id}
```
{head_text}
```

"""
                except:
                    continue
                    
            prompt = f"""### Heads of Files:
{file_heads_text}


You are an advanced language model tasked with summarizing given texts at the beginning of files into a single-line summaries. You are given some pieces of text which are extracted from the heads of files. Please give me the summaries of all the file heads following the instructions below;


### Instructions:
1. **Analyze the File Heads**: Carefully read and understand the provided heads of files.
2. **Guess the Whole Contents **: Based on your analysis and common sense, guess what those files are about.
3. **Create Summaries**: Based on your analysis, create single-line summaries following the output format below.


### Output Format:
'''
(Your analysis and guess)

```json
[
    {{
        "file id":(the id on the top of each file),
        "summary":(single-line summary)
    }},
    ...
]
```
'''


Let's think step by step."""
            
            survey_paths_list.append(survey_paths)
            prompts.append(prompt)

            print("QuickSummary prompt num token: ", SEIMEI.get_num_tokens(prompt))

        llm = LLM(self)
        answers = await llm(prompts)  # to check if the chunk includes core of the simulation

        #print(answers)
        
        outputs = []

        print("survey_paths_list: ", survey_paths_list)
        print("len(survey_paths_list): ", len(survey_paths_list))
        
        for i, answer in enumerate(answers):

            print("i: ", i)
            
            json_text = SEIMEI.extract_text_inside_backticks(answer, "json")

            try:
                json_data = json.loads(json_text)
            except:
                continue
            
            outputs_ = []
            for data in json_data:
                try:
                    id_text = data["file id"]
                    id = int(id_text)
                    
                    data["file_path"] = survey_paths_list[i][id]
                    outputs_.append(data)
                except:
                    traceback.print_exc()

            outputs += outputs_

        return outputs  # [{"file_path":, "summary":}, ...]



