import json, os, re, asyncio
from SEIMEI import SEIMEI, LLM, Job
from AIMO2.EvaluateAnswer import EvaluateAnswer
from AIMO2.SelfCorrection import SelfCorrection
from AIMO2.GiveHint import GiveHint

class MakeAnswer(Job):

    def __init__(self, seimei):
        super().__init__(seimei)
        self.log_off__ = True


    def get_keys(self):
        return {"ids":[], "keys":[]}

    
    async def inference(self, kwargs):

        query, method, steps = kwargs["query"], kwargs["method"], kwargs["steps"]

        step_text = ""
        for i, step in enumerate(steps):
            step_text = f"Step {i+1}: {step}\n\n"
                    
        prompt = f"""<s>[INST]Here you will be given a math problem and a method and inference steps to solve the problem.

Problem: {query}

Method: {method}

{step_text}
Please give me an answer of the problem using the given method and steps.[/INST]"""

        llm = LLM()
        answer = await llm(prompt)

        detailed_analysis = None
        cause = None
        hint = None
        pre_answer = None

        while True:
            kwargs = {"query":query, "answer":answer}
            evaluation = await EvaluateAnswer(self.seimei)(kwargs)
        
            if evaluation["correctness"] == True:
                SEIMEI.correct_answers.append({"query":query, "method": method, "steps": steps, "thought":evaluation["thought"], "answer": answer, "hint":hint, "detailed_analysis":detailed_analysis, "cause": cause, "pre_answer":pre_answer})
                SEIMEI.answer_end = True
                SEIMEI.final_answer = answer
                break
                
            elif evaluation["correctness"] == False:
                SEIMEI.wrong_answers.append({"query":query, "method": method, "steps": steps, "answer": answer})
                
                json_data = await GiveHint(self.seimei)({"query":query, "answer": answer})
                detailed_analysis = json_data["detailed_analysis"]
                cause = json_data["cause"]
                hint = json_data["hint"]

                pre_answer = answer
                answer = await SelfCorrection(self.seimei)({"query":query, "answer": pre_answer, "hint": hint})
                
            else:
                break




