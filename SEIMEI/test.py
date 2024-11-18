from SEIMEI import SEIMEI
import asyncio

processed_path = "./processed"  # input path same as save_path you used in Preparation
expert_class_names = ["Answer", "CheckInf", "MetaSurvey"] # "StructureAnalysis", "ChunkSurvey", "FileSurvey", "MetaSurvey"]
se_restrictions = ["MetaSurvey"]  # search engine only hits classes in this list usually (except when adding expert_restriction in kwargs)
expert_module_names = ["Experts.Code.Modify"]

#seimei = SEIMEI(
#    processed_path = processed_path,
#    expert_class_names = expert_class_names,
#    expert_module_names = expert_module_names,
#    se_restrictions = se_restrictions,
#    max_inference_time = 300,
#    tensor_parallel_size = 1,
#)
