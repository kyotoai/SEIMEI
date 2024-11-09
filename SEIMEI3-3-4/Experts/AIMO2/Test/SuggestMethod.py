import json, os, re
from SEIMEI import SEIMEI, LLM, Job, SearchJob
import inspect, traceback, asyncio
from AIMO2.StepInference import StepInference


class SuggestMethod(Job):

    def __init__(self, seimei):
        super().__init__(seimei)
        self.log_off__ = True


    def get_keys(self):
        key = "This is a job to look for methods to solve a given question"
        return {"ids":[0], "keys":[key]}

    
    async def inference(self, kwargs):

        query = kwargs["query"]

        prompt = f"""<s>[INST]Question: '{query}'

To answer the question above, there are some methods leading to the answer. Please suggest some methods to solve the problem and return them by the following json format:
{{
    "methods": ["method1", "method2", ...]
}}[/INST]"""

        llm = LLM()
        answer = await llm(prompt)

        print("SuggestMethod answer: ", answer)

        # Find the positions of the first '{' and the last '}'
        start_index = answer.find('{')
        end_index = answer.rfind('}')

        methods = []
        
        if start_index != -1 and end_index != -1 and start_index < end_index:
            # Extract the JSON part
            text = answer[start_index:end_index+1]
            
            # Parse the JSON string
            try:
                json_data = json.loads(text)
                methods = json_data["methods"]
            
            except json.JSONDecodeError as e:
                print()
                print(f"Error decoding JSON: {e}")
                print()
                print(text)
                methods = [text]
                
        else:
            print()
            print("No valid JSON found in the text.")
            print()
            print(answer)
            methods = [answer]


        print("SuggestMethod methods: ", methods)
        
        inferences = []
        inference_tasks = []
        for method in methods:
            
            step_inference_instance = StepInference(self.seimei)

            kwargs = {"query":query, "method":method, "steps":[]}
            inferences.append(step_inference_instance(kwargs))

        await asyncio.gather(*inferences)


