import pickle
import os
import numpy as np
curr_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(curr_dir, "AAA_datasets")
path = os.path.join(save_dir, "NSLC_Pretrain_walker_002_10_100.pkl")
# 加载数据
with open(path, "rb") as f:
    data = pickle.load(f)
print(len(data["act_train"]))
for i in range(len(data["act_train"])):
    if np.array_equal(data["act_train"][i],data["act_train"][2]):
        print(i)
# 交互式探索命令
# print(data["act_train"][0])       # 如果是字典
# keys_to_check = ['obs_train', 'act_train', 'obs_val', 'act_val']

# for key in keys_to_check:
#     data = data1[key]
#     if len(data) == 0:
#         print(f"{key}: 数据为空，跳过检查。")
#         continue
#     unique_values, counts = np.unique(data, axis=0, return_counts=True)
    
#     # 统计不同的重复值数量（出现次数>1的唯一值）
#     num_duplicated_values = (counts > 1).sum()
    
#     # 统计总重复实例数（所有重复次数的总和）
#     total_duplicates = (counts - 1).sum()
    
#     print(f"{key}:")
#     print(f"  - 存在 {num_duplicated_values} 个不同的重复值")
#     print(f"  - 总共有 {total_duplicates} 次重复实例\n")