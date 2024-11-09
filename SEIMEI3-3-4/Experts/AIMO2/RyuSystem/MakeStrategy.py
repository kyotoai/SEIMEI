import json, os, re
from SEIMEI import SEIMEI, LLM, Expert
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MakeAnswer2 import MakeAnswer2


class MakeStrategy(Expert):

    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True


    def get_keys(self):
        key = "This is a job to make a problem-solvin strategy to a given problem."
        return {"ids":[0], "keys":[key]}
        #return {"ids":[], "keys":[]}
        

    
    async def inference(self, kwargs):

        query = kwargs["query"]
        
        prompt = f"""<s>[INST]You are a Problem-Solving Strategy Designer. Your task is to develop an effective thinking process for tackling various problems. Always adhere to the following guidelines:

* **Brevity and Focus:** Keep each step concise and focused on essential points.
* **Structure:** Clearly separate each step of the thinking process.
* **Specific Output:** Only output the sections labeled 'Observe,' 'Interpret,' and 'Propose.'  Do not include any introductory or concluding remarks. 
* **Innovation:** Highlight any innovative aspects of your proposed strategies in a brief summary at the end.
* **No Calculations:** Remember, your role is to suggest problem-solving approaches, not to perform calculations or provide a numerical solution.


Develop a problem-solving strategy using the following steps:

**Observe:**
Identify key components and patterns in the given information.
List the relevant mathematical concepts involved.

**Interpret:**
Develop a general approach to solving this type of problem.
Explain the reasoning behind your chosen approach.

**Propose:**
Outline a strategy for tackling the problem, without performing calculations. 


Please make the problem-solving strategy about the following problem:

**Problem:**
{query}[/INST]"""

        llm = LLM(self)
        strategy = await llm(prompt)

        await MakeAnswer2(self)({"query":query, "strategy":strategy})
        

