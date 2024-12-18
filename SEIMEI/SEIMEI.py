import torch
import time, os, re, copy, json, asyncio, ast
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid
import importlib.util, inspect, traceback
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
device = "cuda" if torch.cuda.is_available else "cpu"

log_file_path = "log.json"
if not os.path.exists(log_file_path):
    with open(log_file_path, "w") as json_file:
        json.dump([], json_file)
    logs = []
else:
    with open(log_file_path) as json_file:
        logs = json.load(json_file)


class SEIMEI:
    
    def __init__(self, processed_path = None, model_name = "mistralai/Ministral-8B-Instruct-2410", expert_class_names = None, expert_module_names = None, se_restrictions = None, max_inference_time = 300, tensor_parallel_size = 1, max_seq_len_to_capture = 128000):
        
        SEIMEI.processed_path = processed_path
        SEIMEI.model_name = model_name

        self.job_class_names = expert_class_names
        self.job_module_names = expert_module_names
        self.se_restrictions = se_restrictions
        self.max_inference_time = max_inference_time
        self.tensor_parallel_size = tensor_parallel_size
        self.max_seq_len_to_capture = max_seq_len_to_capture

        # Useful OpenScource LLM Models

        # doesn't require login to huggingface
        # model = "tensorblock/openchat-3.5-0106-GGUF"

        # require login to huggingface
        # model = "google/gemma-2-9b-it"
        # model = "mistralai/Mistral-7B-Instruct-v0.3"
        # model = "mistralai/Ministral-8B-Instruct-2410"
        # model = "meta-llama/Llama-3.1-8B-Instruct"

        engine_args = AsyncEngineArgs(
            model = model_name,
            tensor_parallel_size = tensor_parallel_size,
            max_seq_len_to_capture = max_seq_len_to_capture,
        )
        
        # initialize the engine and the example input
        SEIMEI.engine = AsyncLLMEngine.from_engine_args(engine_args)

        SEIMEI.emb_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").to(device)  # this should be class attribute because self.seimei = seimei in __init__ function in each job duplicates emb_model, which imidiately causes cuda oom

        # These class attributes should be unique to a user question. If these were instance attributes instead, modifying self.seimei from each job would make multiple non-unique instances.
        SEIMEI.infs = []  # [{"inf":, "keep_once":, "keep_permanently":, }]
        SEIMEI.info_dicts = []  # [{"info":str, "query":str, "expert_class_name":str, "expert_instance":inst}, ... ]

        self.called_experts__ = []  # To not get an error when showing Log

        SEIMEI.final_answer = None
        SEIMEI.answer_end = False
        SEIMEI.log_dict = {}

        # for making reinforcement learning dataset
        SEIMEI.correct_answers = []
        SEIMEI.wrong_answers = []

        SEIMEI.queries = []

        SEIMEI.inference_start_time = None

        self.set_jobs()  # job_classes : [str] name list of job class in ./jobs directory



    async def get_answer(self, **kwargs):

        SEIMEI.kwargs = kwargs

        if "query" in kwargs:
            query = kwargs["query"]
        else:
            raise Exception("argument for get_answer must include 'query'")

        # task for being ready for ending inference
        end_inference_task = asyncio.create_task(self.end_inference())

        # task for multi-expert inference
        self.experts = Experts(self)
        expert_task = asyncio.create_task(self.experts(kwargs))

        # task for making log regularly
        save_log_task = asyncio.create_task(self.save_log())

        
        #await permanent_experts_task
        await end_inference_task
        await save_log_task
        #await inference_task

        try:
            await expert_task
        except AnswerEnd:
            pass

        return SEIMEI.final_answer



    async def end_inference(self):

        SEIMEI.inference_start_time = time.time()
        
        while True:
            await asyncio.sleep(5)
            if SEIMEI.answer_end:
                break
            elif time.time() - SEIMEI.inference_start_time > self.max_inference_time:
                print()
                print(time.time() - SEIMEI.inference_start_time, " passed")
                print()
                SEIMEI.answer_end = True
                inf = ""
                for i in range(len(SEIMEI.infs)):
                    inf_to_add = f"**information {i}:** \n{SEIMEI.infs[i]['inf']}\n\n\n"
                    inf += inf_to_add
                SEIMEI.final_answer = f"The answer was not clear but seimei got the following information from the database. \n\n\n{inf}"
                break



    # save log in log.json file regularly
    async def save_log(self):
        try:
            logs.append({})
            
            while True:
                await asyncio.sleep(10)
                log_dict = self.make_log_dict(self.experts)
        
                logs.pop()
                logs.append(log_dict)
                
                with open("log.json", "w") as json_file:
                    json.dump(logs, json_file)
    
                print("log.json updated")

                if SEIMEI.answer_end:
                    raise AnswerEnd

        except AnswerEnd:
            pass

        except Exception as e:
            traceback.print_exc()
            

    # make log dict
    def make_log_dict(self, first_instance):
        return {"expert_class_name":first_instance.__class__.__name__, "args":first_instance.kwargs__, "return":first_instance.output__, "called_experts":self.get_called_jobs(first_instance.called_experts__)}

    # recursive function
    def get_called_jobs(self, called_experts):
        if called_experts == []:
            return []
        else:
            return [{"expert_class_name":called_experts[i].__class__.__name__, "args":called_experts[i].kwargs__, "return":called_experts[i].output__, "called_experts":self.get_called_jobs(called_experts[i].called_experts__)} for i in range(len(called_experts))]



    def set_jobs(self):  # job_classes:[str]

        SEIMEI.job_classes = self.find_class(".")  #[job classes]: all classes in ./job directory

        print()
        print("SEIMEI.job_classes: ", self.job_classes)
        print()
        
        self.job_instances = [] #[job instances]: all instances of classes in self.job_classes

        # following lists are for `global_key_id => job_key, local_key_id, job_id`
        SEIMEI.job_keys = [] #[str]: list of keys to trigger jobs
        SEIMEI.local_key_ids = [] #[int]: list of local ids defined in each job
        SEIMEI.job_class_ids = [] #[id1, id1, ..., id2, ...]: list of job class ids corresponds to self.job_keys
        SEIMEI.job_classes2 = []  #[class1, class1, ..., class2, ...]: list of job classes corresponds to SEIMEI.job_class_ids
        SEIMEI.se_restriction_classes = []
        #self.every_step_job_classes = []

        # new
        SEIMEI.job_keys_embs_dict = {} # {class:torch.tensor}
        SEIMEI.local_key_ids_dict = {} # {class:[int]}
        global_key_ids_dict = {} # {class:[int]}

        # permanent experts
        SEIMEI.permanent_expert_classes = []

        global_id_start = 0
        for i in range(len(self.job_classes)):
            job_instance = self.job_classes[i](self)
            self.job_instances.append(job_instance)

            job_keys_for_instance = job_instance.get_keys()

            # old
            SEIMEI.job_keys += job_keys_for_instance["keys"]
            SEIMEI.local_key_ids += job_keys_for_instance["ids"]
            SEIMEI.job_class_ids += [i for _ in range(len(job_keys_for_instance["keys"]))]
            SEIMEI.job_classes2 += [SEIMEI.job_classes[id] for id in SEIMEI.job_class_ids]

            if self.se_restrictions != None:
                if self.job_classes[i].__name__ in self.se_restrictions:
                    SEIMEI.se_restriction_classes.append(self.job_classes[i])

            # new
            global_ids = [j for j in range(global_id_start, global_id_start+len(job_keys_for_instance["keys"]))]
            global_id_start += len(job_keys_for_instance["keys"])
            global_key_ids_dict[SEIMEI.job_classes[i]] = global_ids
            SEIMEI.local_key_ids_dict[SEIMEI.job_classes[i]] = job_keys_for_instance["ids"]
            #self.job_keys_dict[self.job_classes[i]] = job_keys_for_instance["keys"]

            if hasattr(job_instance, 'permanent_call_wait__'):
                SEIMEI.permanent_expert_classes.append(SEIMEI.job_classes[i])

        
        # This shouldn't be class attribute
        SEIMEI.job_keys_embs = torch.tensor(SEIMEI.emb_model.encode(SEIMEI.job_keys)).to("cpu")


        for job_class in SEIMEI.job_classes:
            SEIMEI.job_keys_embs_dict[job_class] = SEIMEI.job_keys_embs[global_key_ids_dict[job_class]]


    
    # Sub functions

    @staticmethod
    def extract_text_inside_backticks(text, arbitrary_text):
        # Define the pattern to match the text inside ``` that follows the arbitrary text
        pattern = re.compile(r'```{}\s*([\s\S]*?)\s*```'.format(re.escape(arbitrary_text)))
    
        # Search for the pattern in the text
        match = pattern.search(text)
    
        if match:
            return match.group(1).strip()
        else:
            return None

    @staticmethod
    def search(query, topk = 1, expert_restriction = None, prohibited_ids = None):
        if type(query) == str:
            query = [query]
            
        query_embs = torch.tensor(SEIMEI.emb_model.encode(query))

        if expert_restriction==None:
            local_key_ids = SEIMEI.local_key_ids
            job_class_list = SEIMEI.job_classes2
            key_embs = SEIMEI.job_keys_embs

        else:
            job_class_list = []
            local_key_ids = []
            key_embs = torch.tensor([])
            for job_class in expert_restriction:
                if job_class in SEIMEI.job_keys_embs_dict:
                    key_embs = torch.cat((key_embs, SEIMEI.job_keys_embs_dict[job_class]), dim=0)
                    local_key_ids += SEIMEI.local_key_ids_dict[job_class]
                    job_class_list += [job_class for _ in range(len(local_key_ids))]

        
        relevance = torch.matmul(query_embs, key_embs.T)

        if not prohibited_ids == None:

            relevance_values, expert_ids = torch.topk(relevance, k = min(topk + len(prohibited_ids), relevance.shape[1]), dim=1)  # [num_q, min(self.num_expert + len(kwargs["prohibited_ids"]), relevance.shape[1])]
            
            modified_expert_ids = []
            for i in range(len(expert_ids)):
                expert_ids_ids = []
                for j in range(len(expert_ids[i])):
                    if not expert_ids[i,j].item() in prohibited_ids:
                        expert_ids_ids.append(j)
                rest_expert_ids = expert_ids[i][expert_ids_ids]
                _, rest_expert_ids_ids = torch.topk(relevance_values[i][expert_ids_ids], k=min(topk, relevance_values[i][expert_ids_ids].shape[0]), dim=0)
                modified_expert_ids_row = []
                for j in range(len(rest_expert_ids_ids)):
                    modified_expert_ids_row.append(rest_expert_ids[rest_expert_ids_ids[j].item()].item())
                modified_expert_ids.append(modified_expert_ids_row)

            outputs = []
            for i in range(len(modified_expert_ids)):
                output = []
                for j in range(len(modified_expert_ids[i])):
                    new_job_class = job_class_list[modified_expert_ids[i][j]]
                    output.append((new_job_class, local_key_ids[modified_expert_ids[i][j]]))
                outputs.append(output)
            
        else:

            relevance_values, expert_ids = torch.topk(relevance, k = min(topk, relevance.shape[1]), dim=1)  # [num_q, num_relevance]

            outputs = []
            for i in range(len(expert_ids)):
                output = []
                for j in range(len(expert_ids[i])):
                    new_job_class = job_class_list[expert_ids[i,j].item()]
                    outputs.append((new_job_class, local_key_ids[expert_ids[i,j].item()]))

        return outputs  #[[(expert_class, local_key_id), ...], ...] : [len(query), topk]


    @staticmethod
    def search_inf(query, topk = 1):

        if type(query) == str:
            query_ = [query]
        else:
            query_ = query

        inf_texts = []
        # SEIMEI.infs  : [{"inf": ...}]
        for inf_dict in SEIMEI.infs:
            inf_texts.append(str(inf_dict))

        query_embs = torch.tensor(SEIMEI.emb_model.encode(query_))
        inf_embs = torch.tensor(SEIMEI.emb_model.encode(inf_texts))

        relevance = torch.matmul(query_embs, inf_embs.T)

        _, inf_ids = torch.topk(relevance, k = min(topk, relevance.shape[1]), dim=1)  # [num_q, num_relevance]

        if type(query) == str:
            outputs = []
            for j in range(inf_ids[0]):
                inf = SEIMEI.infs[inf_ids[0,j].item()]
                outputs.append(inf)
        else:
            outputs = []
            for i in range(len(inf_ids)):
                output = []
                for j in range(len(inf_ids[i])):
                    inf = SEIMEI.infs[inf_ids[i,j].item()]
                    output.append(inf)
                outputs.append(output)

        return outputs  # [[{"inf":, }, ... ], ... ]



    @staticmethod
    def get_2d_top_mask(values, indices, num_inf = 10):
        
        def reshape_2d_to_1d_with_indices(tensor_2d):
            # Reshape the 2D tensor to 1D
            tensor_1d = tensor_2d.reshape(-1)
            
            # Create a tensor of indices
            rows, cols = tensor_2d.shape
            row_indices = torch.arange(rows).repeat_interleave(cols)
            col_indices = torch.arange(cols).repeat(rows)
            
            # Combine row and column indices
            indices = torch.stack((row_indices, col_indices), dim=1)
            
            return tensor_1d, indices

        relevance_values_1d, reshape_indices = reshape_2d_to_1d_with_indices(values)  # [num_q*num_relevance], [num_q*num_relevance, 2]

        _, topk_1d_indices = torch.topk(relevance_values_1d, k=num_inf)
        #reshape_indices[topk_1d_indices] : [self.max_inf_num, 2]

        reshape_indices = reshape_indices.to(device)
        
        top_mask = torch.zeros(indices.shape, dtype=torch.bool).to(device)
        top_mask[reshape_indices[topk_1d_indices].T[0], reshape_indices[topk_1d_indices].T[1]] = True

        return top_mask
                    

    @staticmethod
    def cut_text(text, num_token = 5000, cut_back = True, add_special_tokens=False):
        
        tokenizer = AutoTokenizer.from_pretrained(SEIMEI.model_name)
        input_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        
        if cut_back: modified_ids = input_ids[-num_token:]
        else: modified_ids = input_ids[:num_token]
            
        return tokenizer.decode(modified_ids)


    @staticmethod
    def get_num_tokens(text, add_special_tokens=False):
        
        tokenizer = AutoTokenizer.from_pretrained(SEIMEI.model_name)
        input_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        
        return len(input_ids)
        

    @staticmethod
    def get_job_class(global_key_id):
        job_class_id = SEIMEI.job_class_ids[global_key_id]
        job_class = SEIMEI.job_classes[job_class_id]
        return job_class


    def find_class(self, directory):
        classes = []
        def get_defined_classes(file_path):
            # Parse the file to get the syntax tree
            with open(file_path, "r") as file:
                tree = ast.parse(file.read(), filename=file_path)
        
            # Collect all class names defined in the file
            defined_classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            
            # Collect all class names imported in the file
            imported_classes = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_classes.add(alias.name.split('.')[0])  # Only take top-level module
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imported_classes.add(alias.name)


            # Dynamically load the module from the file
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Get the actual class objects
            class_objects = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if obj.__module__ == module_name and name in defined_classes and name not in imported_classes:
                    class_objects.append(obj)

            return class_objects

        
        if (self.job_class_names != None) and (self.job_module_names == None):
            for root, _, files in os.walk(directory):
                if not ".ipynb_checkpoints" in root:
                    for file in files:
                        if file.endswith('.py'):
                            file_path = root+"/"+file
                            classes_ = get_defined_classes(file_path)
    
                            for class_ in classes_:
                                if class_.__name__ in self.job_class_names:
                                    classes.append(class_)
                                    
        elif (self.job_class_names == None) and (self.job_module_names != None):
            for root, _, files in os.walk(directory):
                if not ".ipynb_checkpoints" in root:
                    for file in files:
                        if file.endswith('.py'):
                            file_path = root+"/"+file

                            job_in_module_list = False
                            for job_module_name in self.job_module_names:
                                job_module_name = job_module_name.replace(".","/")
                                if job_module_name in file_path:
                                    job_in_module_list = True
                                    break

                            if job_in_module_list:
                                classes_ = get_defined_classes(file_path)
                                for class_ in classes_:
                                    classes.append(class_)

        elif (self.job_class_names != None) and (self.job_module_names != None):
            for root, _, files in os.walk(directory):
                if not ".ipynb_checkpoints" in root:
                    for file in files:
                        if file.endswith('.py'):
                            file_path = root+"/"+file

                            job_in_module_list = False
                            for job_module_name in self.job_module_names:
                                job_module_name = job_module_name.replace(".","/")
                                if job_module_name in file_path:
                                    job_in_module_list = True
                                    break

                            if job_in_module_list:
                                classes_ = get_defined_classes(file_path)
                                for class_ in classes_:
                                    if class_.__name__ in self.job_class_names:
                                        classes.append(class_)
                                    
        else:
            raise Exception("either self.job_class_names or self.job_module_names should be not None at least.")

        return classes




class Expert:
    
    def __init__(self, caller): # self here is Subcalss instance
        self.caller_expert__ = caller
        self.called_experts__ = []
        caller.called_experts__.append(self)

        self.kwargs__ = None
        self.output__ = None

        self.info_dict__ = None

    # kwargs : dict
    async def __call__(self, kwargs):

        self.kwargs__ = kwargs

        if SEIMEI.answer_end:
            raise AnswerEnd

        if not hasattr(self, "log_off__"):
            print()
            print(f"Expert {self.__class__} started")
            print()

        try:
            result = await self.inference(kwargs)

        except AnswerEnd:
            raise AnswerEnd
            
        except Exception as e:
            traceback.print_exc()
            result = None


        if SEIMEI.answer_end:
            raise AnswerEnd

        if not hasattr(self, "log_off__"):
            print()
            print()
            print(f"Expert {self.__class__} ended")
            print()
            print(f"result: {result}")
            print()
            print()

        self.output__ = result
        return result
    


    def set_info(self, info = None, query = None):

        info_dict = {}

        if info:
            info_dict["info"] = info

        if query:
            info_dict["query"] = query

        info_dict["expert_class_name"] = self.__class__.__name__
        info_dict["expert_instance"] = self

        SEIMEI.info_dicts.append(info_dict)
        self.info_dict__ = info_dict

    def reset_info(self):
        SEIMEI.info_dicts = []


    def get_info(self, query = None, topk = 3):

        if len(SEIMEI.info_dicts) == 0:
            return []

        environment = ""  # later

        search_text = f"""### Environment:
{environment}

### Query:
{query}"""

        info_texts = []
        for info_dict in SEIMEI.info_dicts:  # SEIMEI.info_dicts  : [{"info": ...}]
            info_texts.append(str(info_dict))


        query_embs = torch.tensor(SEIMEI.emb_model.encode([search_text]))
        inf_embs = torch.tensor(SEIMEI.emb_model.encode(info_texts))

        relevance = torch.matmul(query_embs, inf_embs.T)

        _, info_ids = torch.topk(relevance, k = min(topk, relevance.shape[1]), dim=1)  # [num_q, num_relevance]

        outputs = []
        for j in range(len(info_ids[0])):
            info = SEIMEI.info_dicts[info_ids[0,j].item()]
            outputs.append(info)

        return outputs  # [{"info":, }, ... ]






class Search(Expert):

    def __init__(self, caller, num_expert = 3):
        super().__init__(caller)
        self.log_off__ = True
        self.num_expert = num_expert # number of called experts per question

    def get_keys(self):
        return {"ids":[], "keys":[]}


    # kwargs : {"queries":str, "job_restriction":[], "avoid_overlapping":str, "prohibited_ids":[]}
    async def inference(self, kwargs):

        if kwargs["queries"] == []:
            return None

        # summarize overlapping queries
        summarize_search_queries = SummarizeSearchQueries(self)
        
        if "avoid_overlapping" in kwargs:
            if kwargs["avoid_overlapping"]:
                kwargs = await summarize_search_queries(kwargs)
        else:  # avoid overlapping in default
            kwargs = await summarize_search_queries(kwargs)

        if kwargs["queries"] == []:
            return None

        query_embs = torch.tensor(SEIMEI.emb_model.encode(kwargs["queries"]))

        if not "job_restriction" in kwargs:
            if SEIMEI.se_restriction_classes == []:
                local_key_ids = SEIMEI.local_key_ids
                job_class_list = SEIMEI.job_classes2
                key_embs = SEIMEI.job_keys_embs
            else:
                job_class_list = []
                local_key_ids = []
                key_embs = torch.tensor([])
                job_classes = SEIMEI.se_restriction_classes
                for job_class in job_classes:
                    if job_class in SEIMEI.job_keys_embs_dict:
                        key_embs = torch.cat((key_embs, SEIMEI.job_keys_embs_dict[job_class]), dim=0)
                        local_key_ids += SEIMEI.local_key_ids_dict[job_class]
                        job_class_list += [job_class for _ in range(len(local_key_ids))]


        else:
            job_class_list = []
            local_key_ids = []
            key_embs = torch.tensor([])
            job_classes = kwargs["job_restriction"]
            for job_class in job_classes:
                if job_class in SEIMEI.job_keys_embs_dict:
                    key_embs = torch.cat((key_embs, SEIMEI.job_keys_embs_dict[job_class]), dim=0)
                    local_key_ids += SEIMEI.local_key_ids_dict[job_class]
                    job_class_list += [job_class for _ in range(len(local_key_ids))]

        
        relevance = torch.matmul(query_embs, key_embs.T)

        if "prohibited_ids" in kwargs:

            relevance_values, expert_ids = torch.topk(relevance, k = min(self.num_expert + len(kwargs["prohibited_ids"]), relevance.shape[1]), dim=1)  # [num_q, min(self.num_expert + len(kwargs["prohibited_ids"]), relevance.shape[1])]
            
            modified_expert_ids = []
            for i in range(len(expert_ids)):
                expert_ids_ids = []
                for j in range(len(expert_ids[i])):
                    if not expert_ids[i,j].item() in kwargs["prohibited_ids"]:
                        expert_ids_ids.append(j)
                rest_expert_ids = expert_ids[i][expert_ids_ids]
                _, rest_expert_ids_ids = torch.topk(relevance_values[i][expert_ids_ids], k=min(self.num_inf, relevance_values[i][expert_ids_ids].shape[0]), dim=0)
                modified_expert_ids_row = []
                for j in range(len(rest_expert_ids_ids)):
                    modified_expert_ids_row.append(rest_expert_ids[rest_expert_ids_ids[j].item()].item())
                modified_expert_ids.append(modified_expert_ids_row)

            outputs = []
            for i in range(len(modified_expert_ids)):
                for j in range(len(modified_expert_ids[i])):
                    new_job_class = job_class_list[modified_expert_ids[i][j]]
                    kwargs = copy.deepcopy(kwargs)
                    kwargs["query"] = kwargs["queries"][i]
                    kwargs["local_key_id"] = local_key_ids[modified_expert_ids[i][j]]
                    outputs.append((kwargs, new_job_class))
            
        else:
            
            relevance_values, expert_ids = torch.topk(relevance, k = min(self.num_expert, relevance.shape[1]), dim=1)  # [num_q, num_relevance]

            outputs = []
            for i in range(len(expert_ids)):
                for j in range(len(expert_ids[i])):
                    new_job_class = job_class_list[expert_ids[i,j].item()]
                    kwargs = copy.deepcopy(kwargs)
                    kwargs["query"] = kwargs["queries"][i]
                    kwargs["local_key_id"] = local_key_ids[expert_ids[i,j].item()]
                    outputs.append((kwargs, new_job_class))

        # outputs : [(kwargs, job_class), ...]
        expert_inferences = [expert_class(self)(kwargs) for kwargs, expert_class in outputs]
        
        await asyncio.gather(*expert_inferences)






class Search2(Expert):  # queries are summed up

    num_inf = 3
    
    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True

    def get_keys(self):
        return {"ids":[], "keys":[]}

    
    async def inference(self, kwargs):  # kwargs : {"queries":, }

        if kwargs["queries"] == []:
            return []

        # summarize overlapping queries
        summarize_search_queries = SummarizeSearchQueries(self)
        
        if "avoid_overlapping" in kwargs:
            if kwargs["avoid_overlapping"]:
                kwargs = await summarize_search_queries(kwargs)
        else:  # avoid overlapping in default
            kwargs = await summarize_search_queries(kwargs)


        query_embs = torch.tensor(SEIMEI.emb_model.encode(kwargs["queries"]))

        if not "job_restriction" in kwargs:
            relevance = torch.matmul(query_embs, SEIMEI.job_keys_embs.T)
            if relevance.shape[1] < self.num_inf: self.num_inf = relevance.shape[1] # to avoid error in topk sentence
            relevance_values, inf_ids = torch.topk(relevance, k = self.num_inf, dim=1)  # [num_q, num_relevance]
            top_inf_mask = self.get_2d_top_mask(relevance_values, inf_ids)

            outputs = []
            for i in range(len(inf_ids)):
                for j in range(len(inf_ids[i])):
                    if top_inf_mask[i,j]:
                        new_job_class = SEIMEI.get_job_class(inf_ids[i,j].item())
                        kwargs = copy.deepcopy(kwargs)
                        kwargs["query"] = kwargs["queries"][i]
                        kwargs["local_key_id"] = SEIMEI.local_key_ids[inf_ids[i,j].item()]
                        outputs.append((kwargs, new_job_class))

        else:
            job_class_list = []
            local_key_ids = []
            key_embs = torch.tensor([])
            job_classes = kwargs["job_restriction"]
            for job_class in job_classes:
                if job_class in SEIMEI.job_keys_embs_dict:
                    key_embs = torch.cat((key_embs, SEIMEI.job_keys_embs_dict[job_class]), dim=0)
                    local_key_ids += SEIMEI.local_key_ids_dict[job_class]
                    job_class_list += [job_class for _ in range(len(local_key_ids))]
                    #job_keys += self.seimei.job_keys_dict[job_class]

            relevance = torch.matmul(query_embs, key_embs.T)
            if relevance.shape[1] < self.num_inf: self.num_inf = relevance.shape[1] # to avoid error in topk sentence
            relevance_values, inf_ids = torch.topk(relevance, k = self.num_inf, dim=1)  # [num_q, num_relevance]
            top_inf_mask = self.get_2d_top_mask(relevance_values, inf_ids)

            outputs = []
            for i in range(len(inf_ids)):
                for j in range(len(inf_ids[i])):
                    if top_inf_mask[i,j]:
                        new_job_class = job_class_list[inf_ids[i,j].item()]
                        kwargs = copy.deepcopy(kwargs)
                        kwargs["query"] = kwargs["queries"][i]
                        kwargs["local_key_id"] = local_key_ids[inf_ids[i,j].item()]
                        outputs.append((kwargs, new_job_class))

        # outputs : [(kwargs, job_class), ...]
        expert_inferences = [expert_class(self)(kwargs) for kwargs, expert_class in outputs]
        
        await asyncio.gather(*expert_inferences)


    # Sub functions

    def get_2d_top_mask(self, values, indices):
        
        def reshape_2d_to_1d_with_indices(tensor_2d):
            # Reshape the 2D tensor to 1D
            tensor_1d = tensor_2d.reshape(-1)
            
            # Create a tensor of indices
            rows, cols = tensor_2d.shape
            row_indices = torch.arange(rows).repeat_interleave(cols)
            col_indices = torch.arange(cols).repeat(rows)
            
            # Combine row and column indices
            indices = torch.stack((row_indices, col_indices), dim=1)
            
            return tensor_1d, indices

        relevance_values_1d, reshape_indices = reshape_2d_to_1d_with_indices(values)  # [num_q*num_relevance], [num_q*num_relevance, 2]

        _, topk_1d_indices = torch.topk(relevance_values_1d, k=self.num_inf)
        #reshape_indices[topk_1d_indices] : [self.max_inf_num, 2]

        reshape_indices = reshape_indices.to(device)
        
        top_mask = torch.zeros(indices.shape, dtype=torch.bool).to(device)
        top_mask[reshape_indices[topk_1d_indices].T[0], reshape_indices[topk_1d_indices].T[1]] = True

        return top_mask


class SummarizeSearchQueries(Expert):
    
    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True

    def get_keys(self):
        return {"ids":[], "keys":[]}

    async def inference(self, kwargs):
        
        past_query_text = ""
        for i, query in enumerate(kwargs["queries"]):
            past_query_text += f"query {i}: {query}\n\n"

        new_query_text = ""
        for i, query in enumerate(SEIMEI.queries):
            new_query_text += f"query {i}: {query}\n\n"


        prompt = f"""<s>[INST]You are an excellent help assistant. You will be provided with two types of queries: a list of past queries and a list of new queries. In the new queries list, there may be some queries that have overlapping meanings with queries in both the new queries list and the past queries list. Your task is to identify and remove those overlapping queries from the new queries list, and then return a modified list of non-overlapping new queries. Note that even if some of the queries are not exactly the same sentence, queries that convey almost the same meaning should be considered as overlapping.


### Past Queries List:
'''
{past_query_text}
'''


### New Queries List:
'''
{new_query_text}
'''


Please return the result in the following JSON format:
{{
    "new_queries": ["(non-overlapping query)", ...]
}}[/INST]"""


        llm = LLM(self)
        output = await llm(prompt)

        try:

            # Find the positions of the first '{' and the last '}'
            start_index = output.find('{')
            end_index = output.rfind('}')

            if start_index != -1 and end_index != -1 and start_index < end_index:
                # Extract the JSON part
                json_text = output[start_index:end_index+1]
                json_text = json_text.replace("\n", "").replace("\\", "")

                json_answer = json.loads(json_text)
                queries = json_answer["queries"]
                json_success = True

        except:
            queries = kwargs["queries"]
            json_success = False

        SEIMEI.queries += queries

        kwargs["queries"] = queries
        kwargs["json_success"] = json_success

        return kwargs



class LLM:

    def __init__(self, caller, max_new_tokens = 4000, max_length = 128000, temperature = 0.0, num_answers = 1):

        self.caller_expert__ = caller
        self.called_experts__ = [] # to avoid error for saving logs
        caller.called_experts__.append(self)
        
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.temperature = temperature
        self.num_answers = num_answers

        self.kwargs__ = None
        self.output__ = None


    async def __call__(self, prompt):

        self.kwargs__ = prompt

        if SEIMEI.answer_end:
            raise AnswerEnd

        if type(prompt) == str:
            
            if self.num_answers == 1:
                output = await self.get_output(prompt)
                self.output__ = output
                return output
                
            else:
                prompts = [prompt for _ in range(self.num_answers)]
                get_outputs = [self.get_output(prompt_) for prompt_ in prompts]
                outputs = await asyncio.gather(*get_outputs)
                self.output__ = outputs
                return outputs
            
        elif type(prompt) == list:
            get_outputs = [self.get_output(prompt_) for prompt_ in prompt]
            outputs = await asyncio.gather(*get_outputs)
            self.output__ = outputs
            return outputs

        else:
            raise Exception("argument for LLM must be either str or list of str")

    
    async def get_output(self, prompt):
        request_id = random_uuid()
        results_generator = SEIMEI.engine.generate(prompt, SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens), request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        return final_output.outputs[0].text


class Experts(Expert):
    def __init__(self, caller):  # caller : None
        super().__init__(caller)

    async def inference(self, kwargs):

        specific_expert_task = asyncio.create_task(SpecificExperts(self)(kwargs))
        permanent_expert_task = asyncio.create_task(PermanentExperts(self)(kwargs))

        await specific_expert_task
        await permanent_expert_task


class SpecificExperts(Expert):
    def __init__(self, caller):  # caller : Experts
        super().__init__(caller)

    async def inference(self, kwargs):
        query = kwargs["query"]
        first_job_instance = Search(self)
        await first_job_instance({"queries":[query]})


class PermanentExperts(Expert):
    def __init__(self, caller):  # caller : Experts
        super().__init__(caller)

    async def inference(self, kwargs):
        permanent_experts = [PermanentExpert(self, inference_class)(kwargs) for inference_class in SEIMEI.permanent_expert_classes]
        await asyncio.gather(*permanent_experts)


class PermanentExpert(Expert):
    def __init__(self, caller, expert_class):  # caller : PermanentExperts
        super().__init__(caller)
        self.expert_class = expert_class

    async def inference(self, kwargs):
        while True:
            expert_instance = self.expert_class(self)
            await expert_instance(kwargs)
            await expert_instance.permanent_call_wait__()


class AnswerEnd(Exception):
    pass

