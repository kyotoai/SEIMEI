import json, os, re
from SEIMEI import SEIMEI, LLM, Expert, Search
import traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SummarizeInf import SummarizeInf
from MetaSurvey2 import MetaSurvey2


class CollectInfoToModifyCode(Expert):

    description = "This expert collects info to modify code files as user query asks."

    num_last_info_dicts = 0
    processed_info_dict_ids = []

    file_paths = []

    def __init__(self, caller, max_inf_tokens = 50000):
        super().__init__(caller)
        self.max_inf_tokens = max_inf_tokens

    def get_keys(self):
        return {"ids":[], "keys":[]}
    

    async def wait(self):
        await asyncio.sleep(10)
    

    async def inference(self, kwargs):

        file_id, instruction = kwargs["file_id"], kwargs["instruction"]

        if len(SEIMEI.info_dicts) <= CollectInfoToModifyCode.num_last_info_dicts:
            return None

        info_id = 0
        info = ""
        for i in range(len(SEIMEI.info_dicts)):
            if i not in CollectInfoToModifyCode.processed_info_dict_ids:
                info_to_add = f"info {info_id}:\n```\n{SEIMEI.info_dicts[i]['info']}\n```\n\n"
                info_id += 1
                info += info_to_add

        if SEIMEI.get_num_tokens(info) > self.max_inf_tokens:
            summarize_info = SummarizeInf(self)
            info = await summarize_info({"inf":info})

        prompt = f"""### INFO:
{info}


### FILE TO MODIFY:
```path
{path}
```

### INSTRUCTION:
```inst
{instruction}
```

### QUERY: 
```query
{SEIMEI.kwargs["query"]}
```


You are an advanced language model tasked with figuring out pieces of information necessary to modify a file as the given query asks. You are given some pieces of information, path of the file to modify, instruction for the modification and user’s query, and you should judge which pieces of information are needed to modify the file following the instruction. Please follow the instructions and output format below.


### Instructions: 
1. **Analyze the Given Info, Instruction and Query**: Carefully understand and analyze them thinking about which pieces of information are relevant to follow the instruction.
2. **Judge**: Based on your analysis, judge which pieces of information are needed. If you don’t find any information relevant to the file modification you shouldn’t designate any piece of it.
3. **Generate Output**: Based on your judgement, return the output following the format below. 


### Output Format:
‘’’
(Your careful analysis and judgement)

```info ids
[ id1 (info id at the beginning of each piece of information), id2, … ]
```
‘’’


Let’s think step by step."""

        llm = LLM(self)
        output = await llm(prompt)


        file_ids_text = SEIMEI.extract_text_inside_backticks(output, "file ids")
        file_ids = json.loads(file_ids_text)

        for id in file_ids:
            if not MetaSurvey2.file_paths[id] in CollectInfoToModifyCode.file_paths:
                CollectInfoToModifyCode.file_paths.append(MetaSurvey2.file_paths[id])

        print("CollectCodeFileToModify file_paths: ", CollectInfoToModifyCode.file_paths)
        
