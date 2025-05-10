import os, json, re, asyncio
import copy
from SEIMEI import SEIMEI, LLM, Expert


class PreprocessFile(Expert):

    description = "This expert split a file content into a small chunk so that llm can process."

    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True

        #self.doc_path = SEIMEI.doc_path
        #self.save_path = save_path
        
        self.rules = [
            {
                #"SUBROUTINE " : 1,
                #"class " : 1,
                "\\\\section*" : 1,
            },
        
            {
                "\\\\subsection*" : 1,
                #"def " : 1,
                #"void " : 1,
            },
        
            {
                #"if " : 1,
                #"end if" : 0,
                "\\\\begin{center}" : 1,
                "\\\\end{gather*}" : 0,
                "\\\\end{align*}" : 0,   
                "\\\\end{equation*}" : 0,
                "\\\\end{enumerate}" : 0,
            },
        
            {
                #"else " : 1,
                #"elif " : 1,
            },
            
        
            {
                "\n\n" : 0,
                "<0x0A><0x0A>" : 0,
                "\x0A\x0A" : 0,
            },
        
            {
                "\n" : 0,
                "<0x0A>" : 0,
                "\x0A" : 0,
            },
        ]
        
        self.file_info = [
            {"folder_path":"", "extensions":[".tex"]},
        #    {"folder_path":"src", "extensions":[".f90"]},
        #    {"folder_path":"run", "extensions":["", ".q"]},
        #    {"folder_path":"lib", "extensions":[".f90"]},
        #    {"folder_path":"", "extensions":[".txt",".md"]},
        ]
        
        self.model_name = SEIMEI.model_name
        self.max_tokens = 3000
        self.min_tokens = 1000

        self.tokenizer = SEIMEI.tokenizer
        

    def get_keys(self):

        return {"ids":[], "keys":[]}


    async def inference(self, kwargs):

        #origin_experts = self.get_origin__(self)
        #query_dict = origin_experts.kwards__["query_dict"]
        #doc_path = query_dict["doc_path"]

        #survey_path = kwargs["survey_path"]

        #with open(f"{SEIMEI.processed_path}/file_paths.json") as json_file:
        #    file_paths = json.load(json_file)
        #with open(f"{SEIMEI.processed_path}/chunks.json") as json_file:
        #    chunks = json.load(json_file)

        if "survey_path" in kwargs:
            query, survey_path = kwargs["query"], kwargs["survey_path"]


        def split_into_chunks(tokenizer, text, max_tokens, min_tokens, rules, debug):
            chunk_list = []
            instruction_list = []
            warnings = []
            split_rule_keys = []
            
            start_id = 0
            tokenized_text = tokenizer(text, return_tensors="pt", add_special_tokens = False)
            num_tokens = len(tokenized_text["input_ids"][0])
            #text_size = len(text)
            rest_text = text
            
            for i in range(int(num_tokens/min_tokens)+1):
                    
                tokenized_text = tokenizer(rest_text, return_tensors="pt", add_special_tokens = False)
                
                if len(tokenized_text["input_ids"][0]) < max_tokens:
                    process_tokenized_text = tokenized_text["input_ids"][0]
                    processed_text = tokenizer.decode(process_tokenized_text, skip_special_tokens=True)
                    chunk_list.append(processed_text)
                    instruction_list.append(processed_text)
                    warnings.append(0)
                    split_rule_keys.append("[END]")
                    break

                process_tokenized_text = tokenized_text["input_ids"][0][:max_tokens]
                process_text1 = tokenizer.decode(process_tokenized_text, skip_special_tokens=True) # process_text1 is only for getting approximate max_tokens of text
                process_text_size = len(process_text1)
                process_text = rest_text[:process_text_size]

                #determine where should be split
                min_split_text = process_text_size
                is_text_split = False
                #warning = 1
                for j in range(len(rules)):
                    for rule_key in rules[j].keys():
                        split_process_text = process_text.split(rule_key)
                        if len(split_process_text) > 1:
                            size_last_split_process_text = len(split_process_text[-1])
                            if (size_last_split_process_text < min_split_text) and (size_last_split_process_text < (1 - min_tokens/max_tokens)*process_text_size) and (size_last_split_process_text!=process_text_size):
                                is_text_split = True
                                split_rule_key = rule_key
                                min_split_text = size_last_split_process_text + len(rule_key) * rules[j][rule_key]

                                if debug:
                                    print("=====")
                                    print("rule_key", rule_key)
                                    print()
                                    print("j: ", j)
                                    print()
                                    print("process_text[:-min_split_text]: ", process_text[:-min_split_text])
                                    print()
                                    print("process_text[-min_split_text:]: ", process_text[-min_split_text:])
                                    print()

                                
                                break # this is supposed to be unnecessary, but I saw some weird thing without this for some reason. This cause must be figured out at some point
                                
                    if is_text_split:
                        #if j < warning_id: warning = 0
                        break
                                
                if is_text_split:
                    processed_text = process_text[:-min_split_text]
                    rest_text = rest_text[len(processed_text):]
                    split_rule_keys.append(rule_key)
                else:
                    processed_text = process_text
                    rest_text = rest_text[len(processed_text):]
                    split_rule_keys.append("")
                
                #processed_tokenized_text = tokenizer(processed_text, return_tensors="pt", add_special_tokens = False)
                #len_processed_text = len(processed_tokenized_text["input_ids"][0])  #this could be more than max_tokens without min sentence, which caused fatal error

                if len(processed_text)==0:
                    break

                chunk_list.append(processed_text)
                #instruction_list.append(processed_text)
                #warnings.append(warning)
                
                #start_id += len_processed_text # taking from process_tokenized_text to prevent the id from getting wrong

            return chunk_list  #, instruction_list, warnings, split_rule_keys


        try:
            with open(survey_path) as f:
                text = f.read()
        except:
            return {"chunks":[]}
            
        debug = False
        
        chunks = split_into_chunks(self.tokenizer, text, self.max_tokens, self.min_tokens, self.rules, debug)

        print("PreprocessFile survey_path: ", survey_path)
        print("PreprocessFile chunks: ", chunks)
        
        return {"chunks":chunks}

