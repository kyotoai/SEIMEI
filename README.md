<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** https://github.com/othneildrew/Best-README-Template/blob/main/BLANK_README.md
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://kyotoai.org">
    <img src="images/SEIMEI_overlook.png" alt="Logo" width="640" height="360">
  </a>

<h3 align="center">SEIMEI</h3>

  <p align="center">
    Search-Enhanced Interface for Multi-Expertise Integration (SEIMEI)
  </p>
  <p align="center">
    SEIMEI ENABLES 1000s OF AGENTS TO INTERACT WITH EACH OTHER!! With highly intelligent search engine, SEIMEI optimizes reasoning steps (with agents) and achieves SOTA results on tasks requiring deep reasoning!!
    <br />
    <a href="https://github.com/kyotoai/SEIMEI/tree/main/demo"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/kyotoai/SEIMEI/tree/main/demo">View Demo</a>
    ·
    <a href="https://github.com/kyotoai/SEIMEI/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/kyotoai/SEIMEI/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

### Search The Best Agent To Make Deep Reasoning
<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->
<br />
<div align="center">
  <img src="images/SEIMEI_example.png" alt="seimei" width="640" height="360">
</div>

<br />
Here's the example of how SEIMEI works. Each agent interacts with LLM and document and makes inference. These inferences are automatically integrated by search engine and gives an answer of question.

<br />
<div align="center">
  <img src="images/SEIMEI_train_example.png" alt="seimei" width="640" height="360">
</div>

<br />
By training search engine, we can optimize the thinking steps like o1 or deepseek-r1!!

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### The Most Intelligent Search Engine
<div align="center">
  <img src="images/Comparison.png" alt="seimei" width="400" height="360">
</div>

Our proprietary search model performs better than semantic embedding model(so called vector search). The graph above shows the result of training our model (3b) and e5-mistral-7b model to search best agents. While the vector search model cannot really retrieve best agents(because problems and agents do not have similar sentences), our proprietary search model can learn what agents are needed to solve a question and retrieve the best ones!!

<a href="https://github.com/kyotoai/SEIMEI/tree/main/demo"><strong>See more details »</strong></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Achieves State Of The Art Result

<div align="center">
  <img src="images/Improvement.png" alt="seimei" width="500" height="360">
</div>

We acheved an improvement of bigcodebench/deepseek-r1 by our search engine!!

<a href="https://github.com/kyotoai/SEIMEI/tree/main/demo"><strong>See more details »</strong></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Applications of SEIMEI

<div align="center">
  <img src="images/Application1.png" alt="seimei" width="640" height="360">
</div>

<div align="center">
  <img src="images/Application2.png" alt="seimei" width="640" height="360">
</div>

<div align="center">
  <img src="images/Application3.png" alt="seimei" width="640" height="360">
</div>

SEIMEI can be applied to make these useful functions!!

<a href="https://github.com/kyotoai/SEIMEI/tree/main/demo"><strong>See more details »</strong></a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* [![vLLM][vllm.ai]][vllm-url]
* [![Hugging Face][huggingface.co]][huggingface-url]
* [OpenAI](https://platform.openai.com/docs/overview)
  
<!--
* [![Next][Next.js]][Next-url]
* [![React][React.js]][React-url]
* [![Vue][Vue.js]][Vue-url]
* [![Angular][Angular.io]][Angular-url]
* [![Svelte][Svelte.dev]][Svelte-url]
* [![Laravel][Laravel.com]][Laravel-url]
* [![Bootstrap][Bootstrap.com]][Bootstrap-url]
* [![JQuery][JQuery.com]][JQuery-url]
-->

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you build SEIMEI on local gpu or rental server gpu.
You can use it by installing seimei using `pip install` or downloading this directory into your local folder.

### Prerequisites

You need to install RMSearch and SEIMEI library on Cuda & PyTorch Environment.
* RMSearch
  ```sh
  git clone https://github.com/kyotoai/RMSearch.git
  cd RMSearch
  pip install -e .
  ```

### Installation

* by `pip install`
  
1. Install SEIMEI (not prepared yet)
   ```sh
   pip install SEIMEI
   ```


* by downloading SEIMEI repository
  
1. Download the repo
   ```sh
   git clone https://github.com/kyotoai/SEIMEI.git
   cd SEIMEI
   pip install -e .
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Here's an usage example using /Experts/Math module. This module answers mathematical questions with brainstorming steps integrated by RMSearch. You can see more examples in /examples/example.ipynb.

### Quick Start

1. Download LLMs in local directory
    ```sh
    cd /workspace
    pip install "huggingface_hub[hf_transfer]"
    pip install hf_transfer
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir ./qwen3b/
    HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download Ray2333/GRM-Llama3.2-3B-rewardmodel-ft --local-dir ./llama3b-rm/
    ```
    
2. Define seimei
    ```py
    from SEIMEI import SEIMEI

    # problems from Kaggle/AIMO2
    problems = [
        "Three airline companies operate flights from Dodola island. Each company has a different schedule of departures. The first company departs every 100 days, the second every 120 days and the third every 150 days. What is the greatest positive integer $d$ for which it is true that there will be $d$ consecutive days without a flight from Dodola island, regardless of the departure times of the various airlines?",
        "Fred and George take part in a tennis tournament with $4046$ other players. In each round, the players are paired into $2024$ matches. How many ways are there to arrange the first round such that Fred and George do not have to play each other? (Two arrangements for the first round are \textit{different} if there is a player with a different opponent in the two arrangements.)",
        "Triangle $ABC$ has side length $AB = 120$ and circumradius $R = 100$. Let $D$ be the foot of the perpendicular from $C$ to the line $AB$. What is the greatest possible length of segment $CD$?",
        "Find the three-digit number $n$ such that writing any other three-digit number $10^{2024}$ times in a row and $10^{2024}+2$ times in a row results in two numbers divisible by $n$.",
        "We call a sequence $a_1, a_2, \ldots$ of non-negative integers \textit{delightful} if there exists a positive integer $N$ such that for all $n > N$, $a_n = 0$, and for all $i \geq 1$, $a_i$ counts the number of multiples of $i$ in $a_1, a_2, \ldots, a_N$. How many delightful sequences of non-negative integers are there?",
        "Let $ABC$ be a triangle with $BC=108$, $CA=126$, and $AB=39$. Point $X$ lies on segment $AC$ such that $BX$ bisects $\angle CBA$. Let $\omega$ be the circumcircle of triangle $ABX$. Let $Y$ be a point on $\omega$ different from $X$ such that $CX=CY$. Line $XY$ meets $BC$ at $E$. The length of the segment $BE$ can be written as $\frac{m}{n}$, where $m$ and $n$ are coprime positive integers. Find $m+n$.",
        "For a positive integer $n$, let $S(n)$ denote the sum of the digits of $n$ in base 10. Compute $S(S(1)+S(2)+\cdots+S(N))$ with $N=10^{100}-2$.",
        """For positive integers $x_1,\ldots, x_n$ define $G(x_1, \ldots, x_n)$ to be the sum of their $\frac{n(n-1)}{2}$ pairwise greatest common divisors. We say that an integer $n \geq 2$ is \emph{artificial} if there exist $n$ different positive integers $a_1, ..., a_n$ such that 
    \[a_1 + \cdots + a_n = G(a_1, \ldots, a_n) +1.\]
    Find the sum of all artificial integers $m$ in the range $2 \leq m \leq 40$.""",
        "The Fibonacci numbers are defined as follows: $F_0 = 0$, $F_1 = 1$, and $F_{n+1} = F_n + F_{n-1}$ for $n \geq 1$. There are $N$ positive integers $n$ strictly less than $10^{101}$ such that $n^2 + (n+1)^2$ is a multiple of 5 but $F_{n-1}^2 + F_n^2$ is not. How many prime factors does $N$ have, counted with multiplicity?",
        "Alice writes all positive integers from $1$ to $n$ on the board for some positive integer $n \geq 11$. Bob then erases ten of them. The mean of the remaining numbers is $3000/37$. The sum of the numbers Bob erased is $S$. What is the remainder when $n \times S$ is divided by $997$?",
    ]

    # Make input to SEIMEI
    queries = []
    for problem in problems:
        queries.append({"query":problem})

    expert_config = [
        {
            "dir_path" : "../Experts/Math", # can be either folder or file
            "start_class" : ["Brainstorming"]
        }
    ]

    # Define seimei object
    seimei = SEIMEI(
        model_name = "/workspace/qwen3b",
        expert_config = expert_config,
        max_inference_time = 1000,
        tensor_parallel_size = 1,
        max_seq_len_to_capture = 10000,
        gpu_memory_utilization = 0.4,
        llm_backend = "vllm",  # set to "openai" to call the OpenAI API instead of vLLM
    )
    ```
    
3. Get answer by seimei
    ```py
    answers = await seimei.get_answer(queries = queries)
    
    print()
    print()
    print(answers)
    ```

### Built-in Agent Demos

SEIMEI ships with two lightweight reference agents under `seimei/agents`. The snippets below show end-to-end runs for each agent with sample questions you can adapt.

#### `code_act`: controlled shell execution

Sample question — *"Run `ls` in the workspace and report the output."*

```python
import asyncio
from seimei import seimei

async def demo_code_act():
    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/code_act.py"}],
        llm_kwargs={"model": "gpt-4o-mini"},
        allow_code_exec=True,
        allowed_commands=["ls", "echo"],
        agent_log_head_lines=1,
        max_tokens_per_question=2000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "You are an execution assistant that never runs unasked commands."},
            {"role": "user", "content": "Run ```bash\nls\n``` and summarize the stdout."},
        ]
    )
    # The code_act reply is stored as the last agent message
    print(result["msg_history"][-2]["content"])

asyncio.run(demo_code_act())
```

#### `web_search`: fast fact gathering

Sample question — *"What are three recent applications of perovskite solar cells?"*

> Requires `pip install duckduckgo_search`.

```python
import asyncio
from seimei import seimei

async def demo_web_search():
    orchestrator = seimei(
        agent_config=[{"file_path": "seimei/agents/web_search.py"}],
        llm_kwargs={"model": "gpt-4o-mini"},
        agent_log_head_lines=2,
        max_tokens_per_question=4000,
    )

    result = await orchestrator(
        messages=[
            {"role": "system", "content": "You gather concise search summaries."},
            {"role": "user", "content": "Search the web for recent applications of perovskite solar cells."},
        ]
    )
    print(result["msg_history"][-2]["content"])

asyncio.run(demo_web_search())
```

### Automatic knowledge accumulation

Set `generate_knowledge=True` when calling the orchestrator to append run retrospectives into a CSV knowledge base:

```python
result = await orchestrator(
    messages=[{"role": "user", "content": "Find clever ways to speed up our ETL pipeline."}],
    generate_knowledge=True,
    save_knowledge_path="seimei_knowledge/knowledge.csv",
    knowledge_prompt_path="seimei/knowledge/prompts/generate_from_runs.md",  # or point at a custom prompt
)
```

The helper `seimei.knowledge.generate_from_runs` analyses the newly created run directory under `seimei_runs/` and appends JSON-normalized rows to the CSV (creating it on first use). The orchestrator reloads the knowledge store so subsequent runs benefit from the fresh guidance. The default retrospection prompt lives at `seimei/knowledge/prompts/generate_from_runs.md`, but you can point `knowledge_prompt_path` at an alternative such as `seimei/knowledge/prompts/excel.md` for domain-specific guidance.

### Using the OpenAI API backend

SEIMEI can run entirely through the OpenAI API, which removes the requirement for a local GPU or vLLM runtime.

1. Install the optional dependencies:
   ```sh
   pip install openai python-dotenv
   ```
2. Provide your API key via environment variable. The project auto-loads a `.env` file when `python-dotenv` is installed:
   ```dotenv
   OPENAI_API_KEY=sk-your-key
   ```
3. Initialize SEIMEI with the OpenAI backend. Pick the model you want to call and (optionally) keep a Hugging Face tokenizer for tasks such as document chunking:
   ```py
   seimei = SEIMEI(
       expert_config = expert_config,
       llm_backend = "openai",
       openai_model = "gpt-5-nano",
       tokenizer_name = "Qwen/Qwen2.5-3B-Instruct",  # optional but recommended for chunking utilities
   )
   ```
   - Pass `openai_api_key="..."` at initialization if you prefer not to use environment variables.
   - Use `openai_base_url="https://..."` when routing requests through Azure OpenAI or a compatible proxy.
4. Call `await seimei.get_answer(...)` exactly as in the GPU setup. Requests are queued using the same scheduling logic as the vLLM backend.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [ ] Search Integration Mechanism for Experts
  - [ ] Permanent Expert
- [ ] Auto Log System in Jupyter Notebook
- [ ] High Precision Code & Textbook RAG

See the [open issues](https://github.com/github_username/repo_name/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Top contributors:

<!--<a href="https://github.com/github_username/repo_name/graphs/contributors">-->
<a href="https://github.com/kyotoai/SEIMEI/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kyotoai/SEIMEI" alt="contrib.rocks image" />
</a>



<!-- LICENSE -->
## License

Distributed under the Apache-2.0 License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT --><!-- [@twitter_handle](https://twitter.com/twitter_handle) -->
## Contact

* Kentaro Seki - seki.kentaro@kyotoai.org

KyotoAI homepage: [https://kyotoai.org](https://kyotoai.org)

Project Link: [https://github.com/kyotoai/SEIMEI](https://github.com/kyotoai/SEIMEI)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [vLLM](https://docs.vllm.ai/en/latest/)
* [Huggingface](https://huggingface.co)
* [Kyoto University Library](https://www.kulib.kyoto-u.ac.jp/mainlib/en/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[vllm.ai]: https://img.shields.io/badge/vLLM-blue
[vllm-url]: https://docs.vllm.ai/en/latest/
[huggingface.co]: https://img.shields.io/badge/huggingface-yellow
[huggingface-url]: https://huggingface.co
[contributors-shield]: https://img.shields.io/github/contributors/kyotoai/SEIMEI.svg?style=for-the-badge
[contributors-url]: https://github.com/kyotoai/SEIMEI/graphs/contributors
[license-shield]: https://img.shields.io/github/license/kyotoai/SEIMEI.svg?style=for-the-badge
[license-url]: https://github.com/kyotoai/SEIMEI/LICENSE.txt

[forks-shield]: https://img.shields.io/github/forks/kyotoai/SEIMEI.svg?style=for-the-badge
[forks-url]: https://github.com/kyotoai/SEIMEI/network/members
[stars-shield]: https://img.shields.io/github/stars/kyotoai/SEIMEI.svg?style=for-the-badge
[stars-url]: https://github.com/kyotoai/SEIMEI/stargazers
[issues-shield]: https://img.shields.io/github/issues/kyotoai/SEIMEI.svg?style=for-the-badge
[issues-url]: https://github.com/kyotoai/SEIMEI/issues
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/kentaro-seki-b12000339
[product-screenshot]: images/screenshot.png

[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
