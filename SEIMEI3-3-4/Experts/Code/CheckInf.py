import json, os, re
from SEIMEI import SEIMEI, LLM, Expert, Search
import traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Answer import Answer
from SummarizeInf import SummarizeInf


class CheckInf(Expert):

    def __init__(self, caller):
        super().__init__(caller)
        self.max_inf_tokens = 4000

    async def permanent_call_wait__(self, **kwargs):
        await asyncio.sleep(10)
        

    def get_keys(self):
        return {"ids":[], "keys":[], "call_every_step":True}

    
    async def inference(self, kwargs):

        inf = ""
        for i in range(len(SEIMEI.infs)):
            inf_to_add = f"**information {i}:** \n{SEIMEI.infs[i]['inf']}\n\n\n"
            inf += inf_to_add

        if SEIMEI.get_num_tokens(inf) > self.max_inf_tokens:
            summarize_inf = SummarizeInf(self)
            await summarize_inf({"inf":inf})

        prompt = f"""<s>[INST]You are an excellent assistant and are adept at investigating a database. You will be provided with one or more pieces of information from the database. Please judge if the informations extrated from the database are enough to answer the user's question in json format.

### USER QUESTION
'{SEIMEI.kwargs["query"]}'

### INFORMATION
'''
{inf}
'''

Please judge if the information above is enough to answer the user's question by true or false with the thought on why you judge. Note that you must add next questions if the information provided is not fully satisfying the requirements to answer the user's question comprehensively. If there is enough information to answer the user's question, please set [] for next_questions. You must answer to these tasks by the following json format,
{{
    "thought":'''(thought of what you think about those informations toward user's questoon and whether the informations are enough to asnwer the user's question)''',
    "judge": (true if the information is enough to answer user's question else false),
    "next_questions":[(list of question or lacking information)]
}}[/INST]"""

        prompt = SEIMEI.cut_text(prompt)

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
                next_questions = json_output["next_questions"]

                print()
                print("-- judge --")
                print(judge)
                print()
                print("-- next_questions --")
                print(next_questions)
                print()
                    
                if judge:
                    print()
                    print("------- !!!! Got an answer ---------")
                    print()
                    print()
                    answer_job = Answer(self)
                    await answer_job({})
                else:
                    search = Search(self)
                    await search({"queries":next_questions})
            
        except Exception as e:
            traceback.print_exc()
            print("json fail")

