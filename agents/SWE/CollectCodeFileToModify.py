import json, os, re
from SEIMEI import SEIMEI, LLM, Expert, Search
import traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from CheckInf import CheckInf
from SummarizeInf import SummarizeInf
from MetaSurvey2 import MetaSurvey2
from CollectInfoToModifyCode import CollectInfoToModifyCode


class CollectCodeFileToModify(Expert):

    description = "This expert collects files to modify as user query asks."

    num_last_info_dicts = 0
    processed_info_dict_ids = []

    file_paths = []

    def __init__(self, caller, max_inf_tokens = 50000):
        super().__init__(caller)
        self.max_inf_tokens = max_inf_tokens

    async def permanent_call_wait__(self, **kwargs):
        await asyncio.sleep(20)

    def get_keys(self):
        return {"ids":[], "keys":[]}

    
    async def inference(self, kwargs):

        if len(SEIMEI.info_dicts) <= CollectCodeFileToModify.num_last_info_dicts:
            return None

        info_id = 0
        info = ""
        for i in range(len(SEIMEI.info_dicts)):
            if i not in CollectCodeFileToModify.processed_info_dict_ids:
                info_to_add = f"info {info_id}:\n```\n{SEIMEI.info_dicts[i]['info']}\n```\n\n"
                info_id += 1
                info += info_to_add

        if SEIMEI.get_num_tokens(info) > self.max_inf_tokens:
            summarize_info = SummarizeInf(self)
            info = await summarize_info({"inf":info})

        prompt = f"""### INFO:
{info}

### META STRUCTURE:
```meta
{SEIMEI.meta_text}
```


### QUERY: 
```query
{SEIMEI.kwargs["query"]}
```


You are an advanced language model tasked with figuring out files to modify as the given query asks. You are given some pieces of information, meta structure of files and user’s query, and you should judge which files to modify as the query demands and return what to modify in the file by analyzing information. Please follow the instructions and output format below.


### Instructions: 
1. **Analyze the Query and Info**: Carefully understand and analyze the provided query and info, thinking about which files to modify as the query demands and what to modify in each file.
2. **Judge**: Based on your analysis, judge which files to modify to answer the query and return the instruction of the modification. If you don’t find any information to tell which files are to be modified for the query, you shouldn’t specify any file.
3. **Generate Output**: Based on your judgement, return the output following the format below. 


### Output Format:
‘’’
(Your careful analysis and judgement)

```json
[
    {{
        "file id":(file id at the beginning of files),
        "instruction":"(instruction about how to modify the file)"
    }},
    ...
]
```
‘’’


Let’s think step by step."""

        llm = LLM(self)
        output = await llm(prompt)

        json_text = SEIMEI.extract_text_inside_backticks(output, "json")

        print("CollectCodeFileToModify json_text: ", json_text)

        try:
            json_data = json.loads(json_text)
    
            expert_inferences = []
            for json_data_ in json_data:
                file_id = json_data_["file id"]
                instruction = json_data_["instruction"]

                expert = CollectInfoToModifyCode(self)
                expert_inferences.append(expert({"file_id":file_id, "instruction":instruction}))
                
        except Exception as e:
            traceback.print_exc()

        await asyncio.gather(*expert_inferences)



