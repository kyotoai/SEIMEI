import json, os, re
from SEIMEI import SEIMEI, LLM, Expert
import inspect, traceback, asyncio


class EvaluateAnswer(Expert):

    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True
        
    def get_keys(self):
        return {"ids":[], "keys":[]}

    
    async def inference(self, kwargs):

        query, answer = kwargs["query"], kwargs["answer"]
        correct_answer = SEIMEI.kwargs["correct_answer"]
                        
        prompt = f"""<s>[INST]Here you will be given a problem, a user's response and the correct answer.


### Problem
'{query}'


### User's Response
'''
{answer}
'''


### Correct Answer
'''
{correct_answer}
'''


I want you to judge whether the user's response is correct answer to the problem or not. You must judge if the user's response is correct or not only by comparing the final answer values of user's response and correct answer. Note that you should reply with the json format below. You need to first figure out what's the final answer for user's response and correct answer respectively and next judge the correctness of user's response.

{{
    "comparison": "(the final answers for user's response and correct answer, and comparison between them)",
    "correctness": true or false
}}[/INST]"""


        llm = LLM(self)
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

                thought = json_data["comparison"]
                correct = json_data["correctness"]

                if correct:
                    print()
                    print("Evaluate Answer")
                    print("prompt")
                    print(prompt)
                    print()
                    print("output")
                    print(output)

                return {"correctness":correct, "thought":thought, "json_success":True}
            
            except json.JSONDecodeError as e:
                print()
                print(f"Error decoding JSON: {e}")
                print()
                print(json_text)
                thought = json_text

        else:
            print()
            print("No valid JSON found in the text.")
            print()
            print(output)
            thought = output

        return {"correctness":None, "thought":thought, "json_success":False}
    

