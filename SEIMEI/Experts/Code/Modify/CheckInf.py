import json, os, re
from SEIMEI import SEIMEI, LLM, Expert, Search
import traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Answer import Answer
from SummarizeInf import SummarizeInf


class CheckInf(Expert):

    def __init__(self, caller, max_inf_tokens = 50000):
        super().__init__(caller)
        self.max_inf_tokens = max_inf_tokens

    async def permanent_call_wait__(self, **kwargs):
        await asyncio.sleep(10)
        

    def get_keys(self):
        return {"ids":[], "keys":[], "call_every_step":True}

    
    async def inference(self, kwargs):

        inf = ""
        for i in range(len(SEIMEI.info_dicts)):
            inf_to_add = f"```information {i} \n{SEIMEI.info_dicts[i]['info']}\n```\n\n"
            inf += inf_to_add

        if SEIMEI.get_num_tokens(inf) > self.max_inf_tokens:
            summarize_inf = SummarizeInf(self)
            inf = await summarize_inf({"inf":inf})

        prompt = f"""<s>[INST]### INFO:
```info
{inf}
```


### QUERY: 
```query
{SEIMEI.kwargs["query"]}
```


You are an excellent assistant and are adept at investigating a database. You will be provided with one or more pieces of information from the database. Please judge if the informations extracted from the database are enough to answer the query. If you find it not enough to answer the query, you should also answer the next query to get information which is need to answer the query.  Please follow the instructions and output format below. 


### Instructions: 
1. **Analyze the Query and Info**: Carefully understand and analyze the provided query and info, thinking about the info includes enough information to answer the query.
2. **Judge**: Based on your analysis, judge if the info includes enough information to answer the query. If there is not enough information to answer the query, think what information is needed to answer the query.
3. **Generate Output**: Based on your analysis, return the output following the format below. 


### Output Format:
Generate the output following the format below.

‘’’
(Your careful and deep analysis)

```judge 
(true if the information is enough to answer user's question; otherwise false)
```

```next query
(next query to get information which is need to answer the query. leave here blank if judge is true)
```
‘’’


Let’s think step by step.[/INST]"""

        prompt = SEIMEI.cut_text(prompt, num_token = 50000)

        llm = LLM(self)
        output = await llm(prompt)


        judge_text = SEIMEI.extract_text_inside_backticks(output, "judge")
        next_query_text = SEIMEI.extract_text_inside_backticks(output, "next query")

        if "true" in judge_text or "True" in judge_text:
            judge = True
        elif "false" in judge_text or "False" in judge_text:
            judge = False
        else:
            judge = False

        if judge:
            print()
            print("------- !!!! Got an answer ---------")
            print()
            print()

            answer_job = Answer(self)
            await answer_job({})

        else:
            try:
                next_query = json.loads(next_query_text)

                search = Search(self)
                await search({"queries":next_query})

            except Exception as e:
                traceback.print_exc()
                raise Exception()
