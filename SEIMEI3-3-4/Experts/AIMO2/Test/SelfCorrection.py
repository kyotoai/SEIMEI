import json, os, re
from SEIMEI import SEIMEI, LLM, Job, SearchJob
import inspect, traceback, asyncio
from transformers import AutoTokenizer


class SelfCorrection(Job):

    def __init__(self, seimei):
        super().__init__(seimei)
        self.log_off__ = True


    def get_keys(self):
        key = "This is a job to correct answer carefully by checking it logically."
        #return {"ids":[0], "keys":[key]}
        return {"ids":[], "keys":[]}
        

    
    async def inference(self, kwargs):

        query, answer = kwargs["query"], kwargs["answer"]

        if "hint" in kwargs:
            hint = kwargs["hint"]
        else: hint = None

        if hint == None:
            prompt = f"""<s>[INST]You will be given a question and solution made by an user.

Question: '{query}'

Solution:
'''
{answer}
'''

There might be an error in the solution above because of lack of understanding of the question. Please logically think over each step of the solution, correct the error, if any, and rewrite the solution.[/INST]"""

        else:
            
            prompt = f"""<s>[INST]You will be given a question, a wrong solution to the question, and a hint helpful to correct the wrong solution.

Question: '{query}'

Wrong Solution:
'''
{answer}
'''

Hint:
'''
{hint}
'''

There is an error in the solution above because of lack of understanding of the question. You are given a hint from a teacher who knows the correct answer. Please logically think over each step of the solution, correct the error, and rewrite the solution. Note that your answer must be well integrated to the hint and[/INST]"""

        llm = LLM()
        rewritten_answer = await llm(prompt)

        return rewritten_answer
        

