import json, os, re
from SEIMEI import SEIMEI, LLM, Expert, Search
import traceback, asyncio



class SummarizeInf(Expert):

    def __init__(self, caller, max_num_tokens = 10000):
        super().__init__(caller)
        self.max_inf_num_tokens = max_num_tokens
        

    def get_keys(self):
        return {"ids":[], "keys":[], "call_every_step":True}

    
    async def inference(self, kwargs):

        inf_id = 0
        inf_dict_id = 0
        inf_dict = {}
        inf = ""
        for i in range(len(SEIMEI.info_dicts)):
            inf_to_add = f"```information {inf_id+1} \n{SEIMEI.info_dicts[i]['info']}\n```\n\n"
            inf_id += 1

            if SEIMEI.get_num_tokens(inf+inf_to_add) > self.max_inf_num_tokens:
                if SEIMEI.get_num_tokens(inf_to_add) > self.max_inf_num_tokens:
                    inf = SEIMEI.cut_text(inf+inf_to_add, num_token = self.max_inf_num_tokens, cut_back = False)
                    inf_dict[inf_dict_id] = inf
                    inf_dict_id += 1
                    inf = ""
                    inf_id = 0
                else:
                    inf_dict[inf_dict_id] = inf
                    inf_dict_id += 1
                    inf = inf_to_add
                    inf_id = 0
            
            else:
                inf += inf_to_add

        if inf != "":
            inf_dict[inf_dict_id] = inf
        

        prompts = []
        for inf_dict_id in inf_dict:
            prompt = f"""<s>[INST]You are an advanced language model tasked with summarizing information in a detailed manner. Below, you will find a user question followed by a block of information. Your task is to summarize the information in as much detail as possible, ensuring that all relevant points are included. However, you should only summarize the parts of the information that are directly relevant to answering the user's question. Do not omit any relevant details from the original information.


### USER QUESTION:
'{SEIMEI.kwargs["query"]}'


### INFORMATION:
'''
{inf_dict[inf_dict_id]}
'''


### INSTRUCTION:
1. Read the user question carefully to understand what specific information is being sought.
2. Review the provided information and identify all sections that are relevant to the user's question.
3. Summarize the relevant sections in detail, ensuring that no important information is lost.
4. Do not include any information that is not directly relevant to the user's question.


### OUTPUT:
Provide a detailed summary of the relevant information below.[/INST]"""

            prompts.append(prompt)


        llm = LLM(self)
        answers = await llm(prompts)


        for answer in answers:
            self.set_info(info = answer, query = SEIMEI.kwargs["query"])

