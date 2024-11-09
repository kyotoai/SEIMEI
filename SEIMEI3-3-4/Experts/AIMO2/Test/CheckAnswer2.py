import json, os, re
from SEIMEI import SEIMEI, LLM, Job
import inspect, traceback, asyncio

# Not used now
class CheckAnswer2(Job):

    def __init__(self, seimei):
        super().__init__(seimei)
        self.log_off__ = True


    def get_keys(self):
        return {"ids":[], "keys":[]}

    
    async def permanent_call_wait__(self, **kwargs):
        await asyncio.sleep(10)

    
    async def inference(self, kwargs):
        
        print()
        print("correct answer num: ", len(SEIMEI.correct_answers))
        print("wrong answer num: ", len(SEIMEI.wrong_answers))
        print()
