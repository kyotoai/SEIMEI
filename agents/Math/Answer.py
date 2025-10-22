import json, os, re
from SEIMEI import SEIMEI, LLM, Expert  #, Search
import inspect, traceback, asyncio

class Answer(Expert):

    description = "This expert will make an answer to user's original question using summarized information from entire inference."

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):
        return {"ids":[], "keys":[]}

    
    async def inference(self, problem, bs_msg_history):

        prompt = f"""Brainstorming:
{bs_msg_history}


You're an excellent mathmatist who solves the problem. Refering to the brainstorming sentence above, solve the problem below.


Problem: {problem}"""

        llm = LLM(self)
        answer = await llm(prompt)

        print()
        print("---- prompt ----")
        print(prompt)

        print()
        print("---- answer ----")
        print(answer)

        SEIMEI.final_answers[self.query_id__] = answer
        SEIMEI.answer_ends[self.query_id__] = True

        print("Answer Ended")
        

