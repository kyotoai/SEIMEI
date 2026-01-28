import pandas as pd

klg_pool = pd.read_csv("exp11_plasma_gkv_v5/knowledge_v6_6_modified.csv")

print("\n---- original ----")
print(klg_pool)

step_list = []

for i in range(len(klg_pool)):
    if i==73:
        step_list.append(None)
        continue
    step_list.append("<2")

klg_pool["step"] = step_list

print("\n---- modified ----")
print(klg_pool)

klg_pool.to_csv("exp11_plasma_gkv_v5/knowledge_v6_6_modified.csv")