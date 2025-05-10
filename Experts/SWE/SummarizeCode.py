import json, os, re, copy
from SEIMEI import SEIMEI, LLM, Expert, Search
import inspect, traceback, asyncio

from ModifyCodeChunk import ModifyCodeChunk

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class SummarizeCode(Expert):

    description = "This expert summarize a code into a very abstract summary."

    def __init__(self, caller):
        super().__init__(caller)
        self.log_off__ = True


    def get_keys(self):

        return {"ids":[], "keys":[]}


    async def inference(self, kwargs):

        chunk = kwargs["chunk"]

        prompt = f"""### CODE SNIPPET:
```code
{chunk}
```


You are an advanced language model tasked with summarizing programming code into very short summary text. You are given a piece of code above and need to analyze it to generate a very short summary that represents the overall structure of the code. Please follow the instructions, summary rule and output format below;


### Instructions:
1. **Analyze the Code**: Carefully read and understand the provided code snippet.
2. **Identify Key Components**: Identify the main components of the code, such as functions, loops and conditionals. Determine how to summarize these components.
3. **Create a Summary**: Based on your analysis, create a summary that contains only important feature of it and illustrates the entire structure of the code.


### Summary Rule:
If you encounter the following features in the code snippet, follow the instruction.

**Parameter Declaration**: Substitute the entire block for parameter declaration by "#Parameter Declaration ..." since this part is not so important.
```example
# Parameter Declaration
# parameters are read and the type are declared 
```

**Function**: Comment out briefly what the function does and omit less important part in the function.
```example
def function():
    # In this function, process 1 and process 2 are executed
    # process1 : ...
    # process2 : ...
```

**Loop, Condition**: Summarize the content to an appropriate length of text based on its size and comment it out.
```example
if (condition):
    # In this condition, process 1 and process 2 are executed
    # process1 : ...
    # process2 : ...
```

**Comment Out**: Summarize them and write down only important part.

**Other Part**: Make it as short as possible focusing on its overall role.


### Output Format:
Generate the output following the format below:

```summary
(answer here)
```


Now, please return summary following the instruction above."""


        llm = LLM(self)
        answer = await llm(prompt)  # to check if the chunk includes core of the simulation

        summary = SEIMEI.extract_text_inside_backticks(answer, "summary")

        return summary