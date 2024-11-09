import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from MetaSurvey import MetaSurvey


class KeywordSearch(Expert):

    def __init__(self, caller):
        super().__init__(caller)


    def get_keys(self):
        key = "This job will search all chunks from database and find all chunks which include a designated keyword."
        
        return {"ids":[0], "keys":[key]}


    async def inference(self, kwargs):
        # ここでsearchJobをmetaに限ってやる必要がある
        # output = self.search_job({"query":query, "job_restriction":[MetaSurvey], "auto_job_set":False}) # {"kwargs_list":[kwargs1, ...], "job_class_list":[job_class1, ...]}

        if "keywords" in kwargs:
            keywords = kwargs["keywords"]

        elif "query" in kwargs:
            prompt = f"""<s>[INST]You're a helpful assistant to answer user's request. You're given the following query;

Query: {kwargs["query"]}

To answer the query, system will search some information in certain database by some keywords. Your job is to give me some inportant keywords to answer the query. Please answer the keywords following the json format;
{{
    "keywords":[(list of the important keywords relevant to the query)]
}}[/INST]"""

            llm = LLM(self)
            answer = await llm(prompt)

            try:
                start_index = answer.find('{')
                end_index = answer.rfind('}')
                
                if start_index != -1 and end_index != -1 and start_index < end_index:
                    json_text = answer[start_index:end_index+1]
                    json_data = json.loads(json_text)
                    keywords = json_data["keywords"]

                else:
                    keywords = []

            except:
                print()
                print("json fail")
                print("answer: ", answer)
                keywords = []
            
            
        else:
            raise Exception("kwargs in KeywordSearch must include either 'keywords' or 'query'")

        
        
        with open(f"/workspace/processed/{SEIMEI.database_name}/chunks.json") as json_file:
            chunks = json.load(json_file)

        match_chunk_ids = []
        for i, chunk in enumerate(chunks):
            for keyword in keywords:
                if keyword in chunk:
                    match_chunk_ids.append(i)
                    break

        output = {}
        output["match_chunk_ids"] = match_chunk_ids

        if "relevance_query" in kwargs and "topk" in kwargs:
            topk, relevance_query = kwargs["topk"], kwargs["relevance_query"]
            query_embs = torch.tensor(SEIMEI.emb_model.encode([relevance_query])).to("cpu")  # [1, emb_dim]
            
            infs = []
            for match_chunk_id in match_chunk_ids:
                infs.append(chunks[match_chunk_id])
                
            inf_embs = torch.tensor(SEIMEI.emb_model.encode(infs)).to("cpu")

            relevance = torch.matmul(query_embs, inf_embs.T)
            values, indices = torch.topk(relevance, k = topk)  # _, [1, topk]

            top_chunk_ids = []
            for id in indices.squeeze():
                top_chunk_ids.append(match_chunk_ids[id])
                
            output["top_chunk_ids"] = top_chunk_ids

        return output
