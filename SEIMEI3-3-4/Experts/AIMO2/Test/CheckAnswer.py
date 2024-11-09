import json, os, re
from SEIMEI import SEIMEI, LLM, Job
import inspect, traceback, asyncio

# Not used now
class CheckAnswer(Job):

    def __init__(self, seimei):
        super().__init__(seimei)
        self.log_off__ = True


    def get_keys(self):
        return {"ids":[], "keys":[]}

    
    async def permanent_call_wait__(self, **kwargs):
        await asyncio.sleep(3)

    
    async def inference(self, kwargs):

        if SEIMEI.answers:
            llms = []
            for answer in SEIMEI.answers:
                if answer["steps"] != []:
                    step_text = ""
                    for i, step in enumerate(steps):
                        step_text = f"Step {i+1}: {step}\n\n"
                        
                    prompt = f"""<s>[INST]Here you will be given a question, a method, and inference steps from an assistant who is generating an answer to the question.

Question: {SEIMEI.query}

Method: {answer["method"]}

{step_text}
I want to check if some step above include the answer to the question. Please return the answer value of the question by following the json format below.

{{
    "answer": "(answer included in some steps above)" or None
}}

Note that if any of the steps doesn't include any answer to the question, please return None for the answer value.[/INST]"""

                    llm = LLM()
                    llms.append(llm(prompt))

            outputs = await asyncio.gather(*llms)

            print()
            print("-- CheckAnswer outputs --")
            print(outputs)

            for output in outputs:
                # Find the positions of the first '{' and the last '}'
                start_index = output.find('{')
                end_index = output.rfind('}')
                
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    # Extract the JSON part
                    json_text = output[start_index:end_index+1]
                    
                    # Parse the JSON string
                    try:
                        json_data = json.loads(json_text)
                        answer = json_data["answer"]
                        if not answer == None:
                            print()
                            print()
                            print("-------- Got an answer --------")
                            print(answer)
                    
                    except json.JSONDecodeError as e:
                        print()
                        print(f"Error decoding JSON: {e}")
                        print()
                        print(json_text)
                        answer = json_text

                else:
                    print()
                    print("No valid JSON found in the text.")
                    print()
                    print(output)
                    answer = output

                    
        

