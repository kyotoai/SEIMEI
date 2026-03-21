import json
from seimei.train import KlgOptimizer

# Example 1:
# Load the bundled SEIMEI-library Q&A dataset (15 rows, easy → hard)
with open("seimei_dataset/default.json", encoding="utf-8") as f:
    dataset = json.load(f)

# Example 2:
# dataset = [
#     {"Question": "What is 2 + 2?",    "CorrectAnswer": "4"},
#     {"Question": "Capital of France?", "CorrectAnswer": "Paris"},
# ]

new_knowledge = KlgOptimizer(
    dataset=dataset,
    optimizer_type="seimei_v1",
    n_sample=1,
    n_epoch=1,
    n_new_klg_per_epoch=3,
    update_klg_threshold=0.1,
    metric="answer_exact_match",
    load_knowledge_path="seimei_knowledge/default.csv",
    save_knowledge_path="seimei_knowledge/improved.csv",
    cache_path="cache.json",
    # seimei constructor args
    llm_config={"model": "gpt-5-nano", "api_key": "..."},
    agent_config={"agents": [...]},
    log_dir="runs/",
)
# new_knowledge is a list of dicts in the same format as default.csv