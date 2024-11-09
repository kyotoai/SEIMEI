import json, os, re
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio
from transformers import AutoTokenizer


class Critic(Expert):

    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True


    def get_keys(self):
        key = "This is a job to correct answer carefully by checking it logically."
        #return {"ids":[0], "keys":[key]}
        return {"ids":[], "keys":[]}
        

    
    async def inference(self, kwargs):

        query, answer = kwargs["query"], kwargs["answer"]

        prompt = f"""<s>[INST]You are a critical evaluator of mathematical problem-solving processes. Your task is to analyze the reasoning behind a provided solution, focusing on recent steps.
**Guidelines:**
* Focus on Recent Steps: Your analysis should focus on the latest two significant logical steps or calculations in the solution process, even if they are within the same numbered step or paragraph.
* Identification of Steps: When identifying the latest and previous steps, quote them exactly as they appear in the solution process. These should be meaningful units of reasoning, such as complete sentences, equations, or key statements—not just numbers or isolated expressions.
* Logical Connection: Assess whether the latest step follows logically from the previous one.
* Accuracy Check: Evaluate the mathematical correctness of the latest step.
* Constructive Criticism: If errors are present, provide a clear explanation of the mistake and suggest a brief fix.
* No Mention of Completion: Do not comment on whether the solution is complete or incomplete.
* Clarity: Clearly show all relevant points in your critique.
* Brevity: If no issues are found, simply state “No issues.”

Based on the current problem-solving process, provide a localized critique of the recent steps:

Problem:
'''{query}'''

Solution Process:
'''{answer}'''

Identification of Latest Step and Previous Step
Latest step: [Quote the latest significant logical step or calculation exactly as it appears in the solution process.]
Previous step: [Quote the immediate preceding logical step or calculation exactly as it appears in the solution process.]
Logical Transition from Previous to Latest Step [Is this step logically derived from the previous one? Are there any gaps in reasoning or implicit assumptions?]
Accuracy of the Current Step [Is the calculation accurate? Are mathematical symbols and rules correctly applied? If you spot an issue, clearly point out where the problem lies (e.g., calculation error, improper operation, logical leap). Provide a brief suggestion for fixing the error.][/INST]"""

        llm = LLM(self)
        critic = await llm(prompt)

        return critic
        

