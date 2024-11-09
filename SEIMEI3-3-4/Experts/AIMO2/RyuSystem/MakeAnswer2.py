import json, os, re, asyncio
from SEIMEI import SEIMEI, LLM, Expert

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from EvaluateAnswer import EvaluateAnswer

class MakeAnswer2(Expert):

    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True


    def get_keys(self):
        return {"ids":[], "keys":[]}

    
    async def inference(self, kwargs):

        query, strategy = kwargs["query"], kwargs["strategy"]

                    
        prompt = f"""<s>[INST]You are a meticulous problem solver. Your task is to start the problem-solving process based on a given strategy.

**Guidelines:**

* **Follow the Strategy:** Adhere to the provided strategy meticulously, addressing each point in order.
* **Explain Alignment:**  Explain how each step of your reasoning aligns with the strategy.
* **Detailed Explanations:** Provide detailed explanations and justifications for your actions.
* **Show Your Work:** Show all relevant calculations and logical steps clearly.
* **Clarify Ambiguity:** If a step in the strategy is unclear, explain your interpretation before proceeding.
* **Relevance:** Ensure that every part of your solution directly contributes to addressing the problem.
* **Logical Flow:** Maintain a logical flow that clearly connects each step of the strategy to your reasoning.
* **Incomplete Strategy:** If you complete the strategy before fully solving the problem, explain how to proceed.


Here are the problem and strategy. Based on the strategy, start the problem-solving process:

Problem:
'''{query}'''

Strategy:
'''{strategy}'''

Note that you should implement the strategy to solve the problem while simultaneously providing detailed reasoning. Show your work, explain your thought process, and include any calculations or logical steps as you go.[/INST]"""

        llm = LLM(self, temperature = 0.1, num_answers = 3)
        answers = await llm(prompt)

        inferences = [EvaluateAnswer(self)({"query":query, "answer":answer}) for answer in answers]
        evaluations = await asyncio.gather(*inferences)

        for i, evaluation in enumerate(evaluations):
            print()
            print()
            print("answer: ")
            print(answers[i])
            
            if evaluation["correctness"] == True:
                
                SEIMEI.correct_answers.append({"query":query, "strategy": strategy, "answer": answers[i]})
                SEIMEI.answer_end = True
                SEIMEI.final_answer = answer
                
            elif evaluation["correctness"] == False:
                SEIMEI.wrong_answers.append({"query":query, "strategy": strategy, "answer": answers[i]})




