import json, os, re
from SEIMEI import SEIMEI, LLM, Expert   #, Search

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Answer import Answer
import inspect, traceback, asyncio

from rmsearch import Search as RMSearch

class Brainstorming(Expert):

    description = "This expert should be run from round 1 to 3. This expert will make brainstorming."

    def __init__(self, caller):
        super().__init__(caller)  # this is needed to record tracks


    def get_keys(self):
        return {"ids":[0], "keys":[self.description]}

    
    async def inference(self, query):

        problem = query

        with open("../Experts/Math/bs_agents.json") as f:
            agents = json.load(f)

        system = f"""You're great mathmatician who solves difficult mathmatical problems. Now you have to make brainstorming about the following problem.

Problem: {problem}

Brainstorm according to the user's request."""

        msg_history = []
        
        for bs_step in range(3):

            search = RMSearch(model_name = "/workspace/llama3b-rm", tensor_parallel_size = 1, pipeline_parallel_size = 1,)
            search_output = await search([problem], agents)
            
            chosen_agent = search_output[0]["keys"][0]["key"]

            llm = LLM(self, return_history=True)
            
            output, msg_history = await llm(chosen_agent, system=system, msg_history=msg_history)

        answer = Answer(self)
        await answer(problem = problem, bs_msg_history = msg_history)
        

