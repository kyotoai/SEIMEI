import os, json, re, asyncio, traceback
from SEIMEI import SEIMEI, LLM, Expert

class MetaSurvey2CheckInfo(Expert):

    description = "This expert checks all the information gotten through inference so far and extract information necessary only for MetaSurvey2."
    
    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True

    def get_keys(self):
        return {"ids":[], "keys":[]}


    async def inference(self, kwargs):

        query = kwargs["query"]

        info_dicts = self.get_info(query, topk = 15)
        if len(info_dicts) == 0:
            return {"info_dicts":[]}

        info_text = ""
        for i, info_dict in enumerate(info_dicts):
            info_text += f"""**information id: {i}**
```
{info_dict["info"]}
```

"""


        prompt = f"""### INFORMATION:
{info_text}

### QUERY:
```query
{query}
```


You are an advanced language model tasked with checking the pieces of information and pick useful ones to investigate a database and get answer to the given query. You are provided with information and query above. Please follow the instructions and output format below.


### Instructions:
1. **Analyze the Information and Query**: Carefully read the provided information and query, and guess how each file works in the system.
2. **Decision**: Based on the your analysis, determine which pieces of information is relevant to answer the query.
3. **Generate Output**: Based on your decision, return the ids on the top of information which you decided is relevant to answer the query following the output format below.


### Output Format:
'''
(Your careful analysis and decision here)

```json
[ id1, ... ]
```
'''


Let’s think step by step."""

        llm = LLM(self)
        output = await llm(prompt)

        json_text = SEIMEI.extract_text_inside_backticks(output, "json")

        try:
            json_data = json.loads(json_text)

            new_info_dicts = []
            for id in json_data:
                new_info_dicts.append(info_dicts[id])

            return {"info_dicts":new_info_dicts}

        except Exception as e:
            traceback.print_exc()





