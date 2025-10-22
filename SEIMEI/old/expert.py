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

