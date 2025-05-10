import json, os, re
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio


class Answer(Expert):

    description = "This expert will make an answer to user's original question using summarized information from entire inference."

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):
        return {"ids":[], "keys":[]}

    
    async def inference(self, kwargs):

        inf = ""
        j = 1
        for i in range(len(SEIMEI.info_dicts)):
            inf_to_add = f"information {j}: {SEIMEI.info_dicts[i]['info']}\n\n\n"
            inf += inf_to_add
            

        prompt = f"""### INFORMATIONS
'''
{inf}
'''


### USER QUESTION
'{SEIMEI.kwargs["query"]}'


You are an excellent assistant and are adept at investigating a database. You are provided with one or more pieces of information above from the database. Please answer the user's question using the information above."""

        prompt = SEIMEI.cut_text(prompt)

        llm = LLM(self)
        answer = await llm(prompt)

        print()
        print("---- prompt ----")
        print(prompt)

        print()
        print("---- answer ----")
        print(answer)

        SEIMEI.final_answer = answer
        SEIMEI.answer_end = True
        

