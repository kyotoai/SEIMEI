import torch
import time, os, re, copy, json, asyncio, ast, warnings
import importlib.util, inspect, traceback
from copy import deepcopy

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.utils import random_uuid
except ImportError:
    AsyncLLMEngine = AsyncEngineArgs = SamplingParams = random_uuid = None

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer
if load_dotenv:
    load_dotenv()

device = "cuda" if torch.cuda.is_available else "cpu"

log_file_path = "log.json"
if not os.path.exists(log_file_path):
    with open(log_file_path, "w") as json_file:
        json.dump([], json_file)
    logs = []
else:
    with open(log_file_path) as json_file:
        logs = json.load(json_file)

from rmsearch import Search as RMSearch


class Expert:
    
    def __init__(self, caller): # self here is Subcalss instance
        self.caller_expert__ = caller
        self.called_experts__ = []
        caller.called_experts__.append(self)

        self.kwargs__ = None
        self.output__ = None

        self.info_dict__ = None

        if hasattr(self, "standby__"):
            self.is_standby_expert__ = True
        else:
            self.is_standby_expert__ = False

        if hasattr(self, "log_off__"):
            self.is_log_off__ = self.log_off__
        else:
            self.is_log_off__ = False

    # kwargs : dict
    async def __call__(self, *args, **kwargs):

        self.args__ = args
        self.kwargs__ = kwargs

        if self.is_standby_expert__:
            while True:  # while loop should be handled here to not let standby continue to run even after AnswerEnd is raised
                exit = await self.standby__(kwargs)  # until standby returns True, standby continues to run loop
                if SEIMEI.answer_ends[self.query_id__]: raise AnswerEnd  # if AnswerEnd is called, standby expert is terminated
                if exit: break

        if SEIMEI.answer_ends[self.query_id__]:
            raise AnswerEnd

        if not self.is_log_off__:
            #print()
            #print(f"Expert {self.__class__} started")
            pass

        try:
            result = await self.inference(*args, **kwargs)

        except AnswerEnd:
            raise AnswerEnd
            
        except Exception as e:
            traceback.print_exc()
            result = None
            raise AnswerEnd


        if SEIMEI.answer_ends[self.query_id__]:
            raise AnswerEnd

        if not self.is_log_off__:
            #print()
            #print(f"Expert {self.__class__} ended")
            #print()
            #print(f"result: {result}")
            #print()
            pass

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

        environment = self.get_env()

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
    


    def get_env(self, depth = 1, inference_code = False):
        # depth : the number of parent expert included in the inference history
        # inference_code : include inference code's raw text if this is true

        env = ""
        expert = self

        for i in range(depth):

            if inference_code:
                env = f"""**Inference History {depth - i}**
* Expert Class: {expert.__class__.__name__}

* Description:
{expert.description}

* Argument:
{str(expert.kwargs__)}

* Inference Code:
```
{inspect.getsource(expert.__class__.inference)}
```

""" + env

            else:
                env = f"""**Inference History {depth - i}**
* Expert Class: {expert.__class__.__name__}

* Description:
{expert.description}

* Argument:
{str(expert.kwargs__)}

""" + env
                
            try:
                expert = expert.caller_expert__
            except:
                return env

        return env

    def get_origin__(self):
        instance = self
        while instance.__class__ != Experts:
            instance = instance.caller_expert__

        return instance

    def get_doc_path__(self):
        origin = self.get_origin__()
        return origin.kwargs__["doc_path"]

    @property
    def query_id__(self):
        return self.get_origin__().query_id





class LLM:

    def __init__(self, caller, return_history=False, max_new_tokens = 10000, max_length = 50000, temperature = 0.0, num_answers = 1):

        self.caller_expert__ = caller
        self.called_experts__ = [] # to avoid error for saving logs
        caller.called_experts__.append(self)
        self.query_id__ = caller.query_id__

        self.return_history = return_history
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.temperature = temperature
        self.num_answers = num_answers

        self.kwargs__ = None
        self.output__ = None


    async def __call__(self, prompt, system=None, msg_history=None):

        self.kwargs__ = prompt

        if SEIMEI.answer_ends[self.query_id__]:
            raise AnswerEnd

        def build_messages(prompt_content):
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            if msg_history:
                messages += deepcopy(msg_history)
            messages.append({"role": "user", "content": prompt_content})
            return messages
        
        if isinstance(prompt, str):

            if self.num_answers == 1:
                messages = build_messages(prompt)
                request_dict = self._build_request(messages)
                output = await self.get_output(request_dict)
                if self.return_history:
                    messages.append({"role": "assistant", "content": output})
                    if system:
                        messages.pop(0)  # to delete system message from message history
                    return output, messages
                return output
                
            else:
                request_dicts = [self._build_request(build_messages(prompt)) for _ in range(self.num_answers)]
                get_outputs = [self.get_output(request_dict) for request_dict in request_dicts]
                outputs = await asyncio.gather(*get_outputs)
                self.output__ = outputs
                return outputs
            
        elif isinstance(prompt, list):
            get_outputs = []
            for prompt_ in prompt:
                messages = build_messages(prompt_)
                request_dict = self._build_request(messages)
                get_outputs.append(self.get_output(request_dict))
            outputs = await asyncio.gather(*get_outputs)
            self.output__ = outputs
            
            return outputs

        else:
            raise Exception("argument for LLM must be either str or list of str")

    
    def _build_request(self, messages):
        backend = getattr(SEIMEI, "llm_backend", "vllm")
        request_dict = {}

        if backend == "vllm":
            if SEIMEI.tokenizer is None:
                raise RuntimeError("Tokenizer is not initialized for llm_backend='vllm'.")
            prompt = SEIMEI.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            if SEIMEI.llm_template is not None:
                prompt = SEIMEI.llm_template(prompt)
            request_dict["prompt"] = prompt
        elif backend == "openai":
            convo = deepcopy(messages)
            if not convo:
                raise ValueError("OpenAI backend requires at least one message to send.")
            if SEIMEI.llm_template is not None:
                convo[-1]["content"] = SEIMEI.llm_template(convo[-1]["content"])
            request_dict["messages"] = convo
            request_dict["prompt"] = convo[-1]["content"]
        else:
            raise ValueError(f"Unsupported llm_backend '{backend}'.")

        return request_dict

    
    async def standby_llm(self):

        #self.request_dict = request_dict
        request_id = SEIMEI.request_id
        SEIMEI.request_id += 1
        
        #SEIMEI.request_ids.insert(0, request_id)
        SEIMEI.request_ids.append(request_id)

        if len(SEIMEI.processing_ids) < SEIMEI.max_request:
            SEIMEI.request_ids.remove(request_id)
            SEIMEI.processing_ids.append(request_id)
        else:
            while True:
                if request_id in SEIMEI.processing_ids:
                    break
                await asyncio.sleep(1)

        return request_id

    
    async def finish_llm(self, request_id):
        if request_id not in SEIMEI.processing_ids:
            raise Exception("There is no request_id in processing_ids")

        SEIMEI.processing_ids.remove(request_id)
        SEIMEI.finished_ids.append(request_id)

        #request_dict = SEIMEI.requests.pop(0)
        if len(SEIMEI.request_ids) != 0:
            next_request_id = SEIMEI.request_ids.pop(0)
            SEIMEI.processing_ids.append(next_request_id)

    
    async def get_output(self, request_dict):

        backend = getattr(SEIMEI, "llm_backend", "vllm")
        prompt = request_dict.get("prompt")

        request_id = await self.standby_llm()

        try:
            if backend == "openai":
                if SEIMEI.openai_client is None:
                    raise RuntimeError("OpenAI client is not initialized.")
                messages = request_dict.get("messages")
                if not messages:
                    raise ValueError("OpenAI backend expects 'messages' in request_dict.")

                response = await SEIMEI.openai_client.chat.completions.create(
                    model=SEIMEI.openai_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                )

                choice = response.choices[0]
                message = getattr(choice, "message", None)
                if message is None and isinstance(choice, dict):
                    message = choice.get("message")
                if message is None:
                    raise RuntimeError("Unexpected response format from OpenAI API.")

                content = getattr(message, "content", None)
                if content is None and isinstance(message, dict):
                    content = message.get("content")

                if isinstance(content, list):
                    text_chunks = []
                    for part in content:
                        if isinstance(part, str):
                            text_chunks.append(part)
                        elif hasattr(part, "text"):
                            text_chunks.append(part.text)
                        elif isinstance(part, dict):
                            text_chunks.append(part.get("text", ""))
                    output = "".join(text_chunks)
                else:
                    output = content if content is not None else ""

            elif backend == "vllm":
                if SEIMEI.engine is None:
                    raise RuntimeError("vLLM engine is not initialized.")

                results_generator = SEIMEI.engine.generate(
                    prompt,
                    SamplingParams(temperature=self.temperature, max_tokens=self.max_new_tokens),
                    request_id
                )

                final_output = None
                async for request_output in results_generator:
                    final_output = request_output

                if final_output is None or not final_output.outputs:
                    raise RuntimeError("vLLM did not return any outputs.")

                output = final_output.outputs[0].text
            else:
                raise ValueError(f"Unsupported llm_backend '{backend}'.")

            self.output__ = output
            return output

        finally:
            await self.finish_llm(request_id)


class SEIMEI:
    
    def __init__(self,
             database_path = None,
             model_name = "/workspace/qwen3b",  #"Qwen/Qwen2.5-3B-Instruct", # for test
             #model_name = "mistralai/Ministral-8B-Instruct-2410", 
             expert_config = [{
                 "dir_path" : "./Experts/Default", # can be either folder or file
             }],
             se_restrictions = None,
             llm_template = None,
             llm_backend = "vllm",
             tokenizer_name = None,
             openai_model = None,
             openai_api_key = None,
             openai_base_url = None,
             max_inference_time = 300, 
             tensor_parallel_size = 1, 
             max_seq_len_to_capture = 50000,
             gpu_memory_utilization=0.4,
             max_request = 20,
        ):
        
        backend = (llm_backend or "vllm").lower()
        if backend not in {"vllm", "openai"}:
            raise ValueError(f"Unsupported llm_backend '{llm_backend}'. Choose from ['vllm', 'openai'].")

        SEIMEI.llm_backend = backend
        SEIMEI.openai_client = None
        SEIMEI.openai_model = openai_model or "gpt-4o-mini"

        SEIMEI.database_path = database_path
        SEIMEI.model_name = model_name

        self.expert_config = expert_config
        self.se_restrictions = se_restrictions
        SEIMEI.llm_template = llm_template
        self.max_inference_time = max_inference_time
        self.tensor_parallel_size = tensor_parallel_size
        self.max_seq_len_to_capture = max_seq_len_to_capture
        self.max_request= max_request

        #SEIMEI.all_requests = AllRequests(max_request)

        # For managing all LLM requests
        SEIMEI.max_request = max_request
        SEIMEI.requests = []
        SEIMEI.request_ids = []
        SEIMEI.request_id = 0
        SEIMEI.results = []
        SEIMEI.finished_ids = []
        SEIMEI.processing_ids = []

        SEIMEI.search = RMSearch(model_name = "/workspace/llama3b-rm", tensor_parallel_size = 1, pipeline_parallel_size = 1,)

        # Useful OpenScource LLM Models

        # doesn't require login to huggingface
        # model = "tensorblock/openchat-3.5-0106-GGUF"

        # require login to huggingface
        # model = "google/gemma-2-9b-it"
        # model = "mistralai/Mistral-7B-Instruct-v0.3"
        # model = "mistralai/Ministral-8B-Instruct-2410"
        # model = "meta-llama/Llama-3.1-8B-Instruct"

        if backend == "vllm":
            if AsyncLLMEngine is None or AsyncEngineArgs is None or SamplingParams is None:
                raise ImportError("vLLM is required for llm_backend='vllm'. Install vllm or choose llm_backend='openai'.")

            engine_args = AsyncEngineArgs(
                model = model_name,
                tensor_parallel_size = tensor_parallel_size,
                pipeline_parallel_size = 1,
                max_seq_len_to_capture = max_seq_len_to_capture,
                max_model_len = max_seq_len_to_capture,
                gpu_memory_utilization = gpu_memory_utilization,
            )
            
            # initialize the engine and the example input
            SEIMEI.engine = AsyncLLMEngine.from_engine_args(engine_args)
        else:
            SEIMEI.engine = None
            if AsyncOpenAI is None:
                raise ImportError("The openai package is required for llm_backend='openai'. Install it with `pip install openai`.")

            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set. Define it in a .env file or pass openai_api_key when initializing SEIMEI.")

            client_kwargs = {"api_key": api_key}
            if openai_base_url:
                client_kwargs["base_url"] = openai_base_url

            SEIMEI.openai_client = AsyncOpenAI(**client_kwargs)

        tokenizer_target = tokenizer_name or (model_name if backend == "vllm" else None)
        if tokenizer_target:
            SEIMEI.tokenizer = AutoTokenizer.from_pretrained(tokenizer_target, padding_side='left')
        else:
            SEIMEI.tokenizer = None

        # SEIMEI.emb_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1").to(device)  # this should be class attribute because self.seimei = seimei in __init__ function in each expert duplicates emb_model, which imidiately causes cuda oom

        # These class attributes should be unique to a user question. If these were instance attributes instead, modifying self.seimei from each expert would make multiple non-unique instances.
        SEIMEI.infs = []  # [{"inf":, "keep_once":, "keep_permanently":, }]
        SEIMEI.info_dicts = []  # [{"info":str, "query":str, "expert_class_name":str, "expert_instance":inst}, ... ]

        self.called_experts__ = []  # To not get an error when showing Log

        #SEIMEI.final_answer = None
        #SEIMEI.answer_end = False
        SEIMEI.final_answers = []
        SEIMEI.answer_ends = []
        
        SEIMEI.log_dict = {}

        # for making reinforcement learning dataset
        SEIMEI.correct_answers = []
        SEIMEI.wrong_answers = []

        SEIMEI.queries = []

        SEIMEI.data = {}  # { expert_class1:{}, ... }

        SEIMEI.inference_start_time = None

        self.set_experts()  # expert_classes : [str] name list of expert class in ./Experts directory



    async def get_answer(self, **kwargs):

        SEIMEI.kwargs = kwargs

        if not ("query" in kwargs or "queries" in kwargs):
            raise Exception("argument for get_answer must include 'query'")
        
        queries = kwargs["queries"]
        SEIMEI.queries = queries

        SEIMEI.final_answers = [None for _ in range(len(queries))]
        SEIMEI.answer_ends = [False for _ in range(len(queries))]
        

        #save_log_task = asyncio.create_task(self.save_log())

        # task for being ready for ending inference
        end_inference_task = asyncio.create_task(self.end_inference())

        # task for multi-expert inference
        #self.experts = Experts(self)
        #expert_task = asyncio.create_task(self.experts(kwargs))

        # task for making log regularly
        

        await asyncio.gather(
            *[self.expert_tasks(query_dict, query_id) for query_id, query_dict in enumerate(queries)]
        )
        
        #await permanent_experts_task
        await end_inference_task
        
        #await inference_task

        #await save_log_task

        return SEIMEI.final_answers


    async def expert_tasks(self, query_dict, query_id):
        
        # Need to modify save_log and this
        experts = Experts(self, query_id)
        expert_task = asyncio.create_task(experts(**query_dict))
        
        try:
            await expert_task
        except AnswerEnd:
            SEIMEI.answer_ends[query_id] = True


    async def end_inference(self):

        SEIMEI.inference_start_time = time.time()
        
        while True:
            await asyncio.sleep(5)
            if all(SEIMEI.answer_ends):
                break
            elif time.time() - SEIMEI.inference_start_time > self.max_inference_time:
                print()
                print(time.time() - SEIMEI.inference_start_time, " passed")
                print()
                for i in range(len(SEIMEI.answer_ends)):
                    SEIMEI.answer_ends[i] = True
                #inf = ""
                #for i in range(len(SEIMEI.info_dicts)):
                #    inf_to_add = f"**information {i}:** \n{SEIMEI.info_dicts[i]['info']}\n\n\n"
                #    inf += inf_to_add
                #SEIMEI.final_answer = f"The answer was not clear but seimei got the following information from the database. \n\n\n{inf}"
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
        return {"expert_class_name":first_instance.__class__.__name__, "args":first_instance.kwargs__, "return":first_instance.output__, "called_experts":SEIMEI.get_called_experts(first_instance.called_experts__)}

    # recursive function
    @staticmethod
    def get_called_experts(called_experts):
        if called_experts == []:
            return []
        else:
            return [{"expert_class_name":called_experts[i].__class__.__name__, "args":SEIMEI.convert_to_string(called_experts[i].kwargs__), "return":SEIMEI.convert_to_string(called_experts[i].output__), "called_experts":SEIMEI.get_called_experts(called_experts[i].called_experts__)} for i in range(len(called_experts))]

    @staticmethod
    def convert_to_string(obj):
        if isinstance(obj, dict):
            return {str(key): SEIMEI.convert_to_string(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [SEIMEI.convert_to_string(element) for element in obj]
        else:
            return str(obj)


    def set_experts(self):

        #SEIMEI.expert_classes = self.get_expert_classes()  # [ expert_class1, ... ] : all expert classes designated by self.expert_config
        SEIMEI.start_expert_classes = self.get_expert_classes(key="start_class")

        print()
        print("SEIMEI.start_expert_classes: ", self.start_expert_classes)
        print()

        """

        if self.expert_classes == []:
            raise Exception("There is no expert to make inference with.")
        
        # following lists are for `global_key_id => expert_key, local_key_id, expert_id`
        expert_keys = [] #[str]: list of keys to trigger experts
        expert_class_ids = [] #[id1, id1, ..., id2, ...]: list of expert class ids corresponds to self.expert_keys
        SEIMEI.expert_classes2 = []  #[class1, class1, ..., class2, ...]: list of expert classes corresponds to SEIMEI.expert_class_ids
        SEIMEI.se_restriction_classes = []
        SEIMEI.expert_keys_embs_dict = {} # {class:torch.tensor}
        SEIMEI.local_key_ids_dict = {} # {class:[int]}
        global_key_ids_dict = {} # {class:[int]}

        # permanent experts
        SEIMEI.permanent_expert_classes = []

        global_id_start = 0
        for i in range(len(self.expert_classes)):
            expert_instance = self.expert_classes[i](self)

            if hasattr(expert_instance, "get_keys"):
                expert_keys_for_instance = expert_instance.get_keys()
            elif hasattr(expert_instance, "description"):
                expert_keys_for_instance = {"keys":[expert_instance.description], "ids":[0]}
            else:
                continue
                
            expert_keys += expert_keys_for_instance["keys"]
            expert_class_ids += [i for _ in range(len(expert_keys_for_instance["keys"]))]

            global_ids = [j for j in range(global_id_start, global_id_start+len(expert_keys_for_instance["keys"]))]
            global_id_start += len(expert_keys_for_instance["keys"])
            global_key_ids_dict[SEIMEI.expert_classes[i]] = global_ids
            SEIMEI.local_key_ids_dict[SEIMEI.expert_classes[i]] = expert_keys_for_instance["ids"]
            SEIMEI.expert_classes2 += [SEIMEI.expert_classes[id] for id in expert_class_ids]
            if self.se_restrictions != None:
                if self.expert_classes[i].__name__ in self.se_restrictions:
                    SEIMEI.se_restriction_classes.append(self.expert_classes[i])
            

            if hasattr(expert_instance, 'permanent_call_wait__'):
                SEIMEI.permanent_expert_classes.append(SEIMEI.expert_classes[i])

        
        # This shouldn't be class attribute
        expert_keys_embs = torch.tensor(SEIMEI.emb_model.encode(expert_keys)).to("cpu")

        for expert_class in SEIMEI.expert_classes:
            SEIMEI.expert_keys_embs_dict[expert_class] = expert_keys_embs[global_key_ids_dict[expert_class]]
        """
    
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

    '''
    @staticmethod
    def search(query, topk = 1, expert_restriction = None, prohibited_ids = None):
        if type(query) == str:
            query = [query]
            
        query_embs = torch.tensor(SEIMEI.emb_model.encode(query))

        if expert_restriction==None:
            for expert_class in SEIMEI.expert_keys_embs_dict:
                key_embs = torch.cat((key_embs, SEIMEI.expert_keys_embs_dict[expert_class]), dim=0)
                local_key_ids += SEIMEI.local_key_ids_dict[expert_class]
                expert_class_list += [expert_class for _ in range(len(local_key_ids))]

        else:
            expert_class_list = []
            local_key_ids = []
            key_embs = torch.tensor([])
            for expert_class in expert_restriction:
                if expert_class in SEIMEI.expert_keys_embs_dict:
                    key_embs = torch.cat((key_embs, SEIMEI.expert_keys_embs_dict[expert_class]), dim=0)
                    local_key_ids += SEIMEI.local_key_ids_dict[expert_class]
                    expert_class_list += [expert_class for _ in range(len(local_key_ids))]

        
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
                    new_expert_class = expert_class_list[modified_expert_ids[i][j]]
                    output.append((new_expert_class, local_key_ids[modified_expert_ids[i][j]]))
                outputs.append(output)
            
        else:

            relevance_values, expert_ids = torch.topk(relevance, k = min(topk, relevance.shape[1]), dim=1)  # [num_q, num_relevance]

            outputs = []
            for i in range(len(expert_ids)):
                output = []
                for j in range(len(expert_ids[i])):
                    new_expert_class = expert_class_list[expert_ids[i,j].item()]
                    outputs.append((new_expert_class, local_key_ids[expert_ids[i,j].item()]))

        return outputs  #[[(expert_class, local_key_id), ...], ...] : [len(query), topk]
    '''

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
    

    def get_expert_classes(self, key = "class_names"):

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

    
        for config_dict in self.expert_config:

            if os.path.isfile(config_dict["dir_path"]):
                classes_ = get_defined_classes(config_dict["dir_path"])
                for class_ in classes_:
                    if not key in config_dict:
                        classes.append(class_)
                    elif class_.__name__ in config_dict[key]:
                        classes.append(class_)

            else:
                for root, _, files in os.walk(config_dict["dir_path"]):
                    if not ".ipynb_checkpoints" in root:
                        for file in files:
                            if file.endswith('.py'):
                                file_path = root+"/"+file
                                classes_ = get_defined_classes(file_path)
        
                                for class_ in classes_:
                                    if not key in config_dict:
                                        classes.append(class_)
                                    elif class_.__name__ in config_dict[key]:
                                        classes.append(class_)

        return classes


class Experts(Expert):
    def __init__(self, caller, query_id):  # caller : None
        super().__init__(caller)
        self.query_id = query_id

    async def inference(self, *args, **kwargs):

        specific_expert_task = asyncio.create_task(SpecificExperts(self)(*args, **kwargs))
        #permanent_expert_task = asyncio.create_task(PermanentExperts(self)(kwargs))

        await specific_expert_task
        #await permanent_expert_task


class SpecificExperts(Expert):
    def __init__(self, caller):  # caller : Experts
        super().__init__(caller)

    async def inference(self, *args, **kwargs):
        print(args, kwargs)
        if "query" in kwargs:
            inference_functions = []
            for first_expert_class in SEIMEI.start_expert_classes:
                first_expert_instance = first_expert_class(self)
                inference_functions.append(first_expert_instance(*args, **kwargs))
            await asyncio.gather(*inference_functions)
        else:
            raise Exception("kwargs for SpecificExperts must include either query or queries")


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
