import json, os, re
from SEIMEI import SEIMEI, LLM, Expert, Search
import traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from SummarizeInf2 import SummarizeInf2

# This task checks some pieces of information for a query are enough to answer the query. if not, it will seek for more information from the database.
class CheckInf2(Expert):

    def __init__(self, caller):
        super().__init__(caller)
        self.max_inf_tokens = 4000
        

    def get_keys(self):
        return {"ids":[], "keys":[], "call_every_step":True}

    
    async def inference(self, kwargs):

        query, infs, key_ids = kwargs["query"], kwargs["infs"], kwargs["local_key_ids"]

        inf = ""
        for i in range(len(infs)):
            inf_to_add = f"**information {i}:** \n{infs[i]}\n\n\n"
            inf += inf_to_add

        if SEIMEI.get_num_tokens(inf) > self.max_inf_tokens:
            summarize_inf = SummarizeInf(self)
            await summarize_inf({"inf":inf})

        prompt = f"""<s>[INST]You are an excellent assistant and are adept at investigating a database. You will be provided with one or more pieces of information from the database. Please judge if the informations extrated from the database are enough to answer the user's question in json format.

### USER QUESTION
'{query}'

### INFORMATION
'''
{inf}
'''

Please judge if the information above is enough to answer the user's question by true or false with the thought on why you judge. You must answer to these tasks by the following json format,
{{
    "thought":'''(thought of what you think about those informations toward user's questoon and whether the informations are enough to asnwer the user's question)''',
    "judge": (true if the information is enough to answer user's question else false)
}}[/INST]"""

        llm = LLM(self)
        output = await llm(prompt)
        
        try:
            # Find the positions of the first '{' and the last '}'
            start_index = output.find('{')
            end_index = output.rfind('}')
            
            if start_index != -1 and end_index != -1 and start_index < end_index:
                # Extract the JSON part
                json_text = output[start_index:end_index+1]
                json_text = json_text.replace("\n", "").replace("\\", "")
                
                json_output = json.loads(json_text)
                judge = json_output["judge"]
                    
                if judge:
                    sum_out = SummarizeInf2(self)(kwargs)
                    for inf in sum_out["infs"]:
                        SEIMEI.infs.append({"query":query, "inf":inf})
                else:
                    if "prohibited_ids" in kwargs:
                        kwargs["prohibited_ids"] += key_ids
                    else:
                        kwargs["prohibited_ids"] = key_ids
                    search = Search(self)
                    await search(kwargs)
        
        except Exception as e:
            traceback.print_exc()

