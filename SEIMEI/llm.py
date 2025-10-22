
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
                    #temperature=self.temperature,
                    max_completion_tokens=self.max_new_tokens,
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
