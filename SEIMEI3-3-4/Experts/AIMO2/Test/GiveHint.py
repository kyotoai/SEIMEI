import json, os, re
from SEIMEI import SEIMEI, LLM, Job, SearchJob
import inspect, traceback, asyncio
from transformers import AutoTokenizer


class GiveHint(Job):

    def __init__(self, seimei):
        super().__init__(seimei)
        self.log_off__ = True


    def get_keys(self):
        key = "This is a job to correct answer carefully by checking it logically."
        #return {"ids":[0], "keys":[key]}
        return {"ids":[], "keys":[]}
        

    
    async def inference(self, kwargs):

        query, answer = kwargs["query"], kwargs["answer"]
        correct_answer = SEIMEI.kwargs["correct_answer"]

        prompt = f"""<s>[INST]You are a math teacher and helps a student get a correct answer. You will be given a question, student's answer to it, and the correct answer of it.

Question: '{query}'


Student's answer:
'''
{answer}
'''


Correct Answer:
'''
{correct_answer}
'''


The answer from the student had something wrong and couldn't reach the correct answer. You should analyze the ultimate cause of the error and give him a hint correcting the cause and leading him to the correct answer. Note that you shouldn't include the final answer in the hint. Please return the detailed analysis, the cause and the hint by following the json format below;

{{
    "detailed analysis": "(a detailed analysis of why student's answer was wrong and where is the point it started to go different direction)",
    "cause": "(an ultimate cause of the error which makes the student's answer different from the correct answer)",
    "hint": "(a hint correcting the cause of the error and leading him to the correct answer)"
}}[/INST]"""

        llm = LLM()
        output = await llm(prompt)

        # Find the positions of the first '{' and the last '}'
        start_index = output.find('{')
        end_index = output.rfind('}')
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            # Extract the JSON part
            json_text = output[start_index:end_index+1]

            json_text = json_text.replace("\n", "").replace("\\", "")
            
            # Parse the JSON string
            try:
                json_data = json.loads(json_text)

                detailed_analysis = json_data["detailed analysis"]
                cause = json_data["cause"]
                hint = json_data["hint"]

                return {"detailed_analysis":detailed_analysis, "cause":cause, "hint":hint, "json_success":True}
            
            except json.JSONDecodeError as e:
                print()
                print(f"Error decoding JSON: {e}")
                print()
                print(json_text)

                detailed_analysis = json_text
                cause = json_text
                hint = json_text

        else:
            print()
            print("No valid JSON found in the text.")
            print()
            print(output)
            
            detailed_analysis = output
            cause = output
            hint = output

        return {"detailed_analysis":detailed_analysis, "cause":cause, "hint":hint, "json_success":False}
        

