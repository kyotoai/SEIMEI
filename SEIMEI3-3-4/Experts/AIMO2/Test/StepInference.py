import json, os, re
from SEIMEI import SEIMEI, LLM, Job, SearchJob
import inspect, traceback, asyncio
from transformers import AutoTokenizer
from AIMO2.MakeAnswer import MakeAnswer


class StepInference(Job):

    def __init__(self, seimei):
        super().__init__(seimei)
        self.log_off__ = True


    def get_keys(self):
        key = "This is a job to make an inference step forward."
        #return {"ids":[0], "keys":[key]}
        return {"ids":[], "keys":[]}
        

    
    async def inference(self, kwargs):

        query, method, steps = kwargs["query"], kwargs["method"], kwargs["steps"]

        if steps == []:
            prompt = f"""<s>[INST]Question: '{query}'

Method: {method}

To answer the question above, you should proceed the inference a step forward. I need a several cadidates for the next step inference. Please return a list of next one step inferences by following the json format below;
{{
    "candidate list for next step": ["candidate1", "candidate2", ...]
}}[/INST]"""

        else:
            step_text = ""
            for i, step in enumerate(steps):
                step_text = f"Step {i+1}: {step}\n\n"
    
            prompt = f"""<s>[INST]Question: '{query}'

Method: {method}

{step_text}
To answer the question above, you should proceed the inference a step forward. I need a several cadidates for the next step inference. Please return a list of next one step inferences by following the json format below;
{{
    "candidate list for next step": ["candidate1", "candidate2", ...]
}}[/INST]"""

        llm = LLM()
        answer = await llm(prompt)


        # Find the positions of the first '{' and the last '}'
        start_index = answer.find('{')
        end_index = answer.rfind('}')

        next_steps = []
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            # Extract the JSON part
            text = answer[start_index:end_index+1]
            
            # Parse the JSON string
            try:
                json_data = json.loads(text)
                next_steps = json_data["candidate list for next step"]
            
            except json.JSONDecodeError as e:
                print()
                print(f"Error decoding JSON: {e}")
                print()
                print(text)
                next_steps = [text]
                
        else:
            print()
            print("No valid JSON found in the text.")
            print()
            print(answer)
            next_steps = [answer]

        inferences = []
        for next_step in next_steps:
            steps_ = steps + [next_step]
            #print("StepInference steps_: ", steps_)
            kwargs = {"query":query, "method":method, "steps": steps_}
            if len(steps_) >= 2:
                inferences.append(MakeAnswer(self.seimei)(kwargs))
            else:
                inference = self(kwargs)
                inferences.append(inference)

        await asyncio.gather(*inferences)
        

