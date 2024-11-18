from SEIMEI import SEIMEI
import asyncio

processed_path = "./processed"  # input path same as save_path you used in Preparation
expert_class_names = ["Answer", "CheckInf", "MetaSurvey"] # "StructureAnalysis", "ChunkSurvey", "FileSurvey", "MetaSurvey"]
se_restrictions = ["MetaSurvey"]  # search engine only hits classes in this list usually (except when adding expert_restriction in kwargs)
expert_module_names = ["Experts.Code.Modify"]

test_text = "Hello, my name is SEIMEI!"
SEIMEI.model_name = "gpt2"
num_token = SEIMEI.get_num_tokens(test_text)
print(num_token)
