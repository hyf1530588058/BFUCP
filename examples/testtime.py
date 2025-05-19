import random
import numpy as np
import time
import torch
import os
# from ppo.myPPOmodel2 import Policy
from ppo.metaPPOmodel import Policy
from thop import profile
from ppo.arguments import get_args
import sys
import itertools
import torch.nn as nn
from ppo.hgtmodel import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
curr_dir = os.path.dirname(os.path.abspath(__file__))  #打印当前文件的绝对路径,获取当前文件上一层目录#
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)   #定义搜索路径的优先级顺序#
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure
structures = []
for i in range(1):
    save_path_structure = os.path.join(curr_dir,"robot_universal/walker",str(i) + ".npz")
    np_data = np.load(save_path_structure)    #读取文件
    structure_data = []
    for key, value in itertools.islice(np_data.items(), 2):  #将读取的原数据添加到新的预训练列表中#
        structure_data.append(value)
    structure_data = tuple(structure_data)           
    structure = (structure_data[0], structure_data[1])
batch_size = 1
input_dim = 202
inputs = torch.randn(batch_size, input_dim).to(device)

args = get_args()    #在训练最外层初始化模型#
actor_critic = Policy(
    base_kwargs={'recurrent': args.recurrent_policy})
actor_critic.to(device)
actor_critic.eval()

times = []
warmup_steps = 10
test_steps = 100

# 预热
for _ in range(warmup_steps):
    _ = actor_critic.act(structure, inputs, None, None, None)

# 正式测试
for _ in range(test_steps):
    start = time.time()
    _ = actor_critic.act(structure, inputs, None, None, None)
    times.append((time.time() - start) * 1000)  # 转换为毫秒

avg_time = np.mean(times)
std_time = np.std(times)
print(f"平均推理时间: {avg_time:.3f}ms ± {std_time:.3f}ms")
def count_hgt_flops(m, x, y):
    # HGTConv的FLOPs估算
    # x[0]: node_features, x[1]: WW, x[2]: mask_matrix, etc.
    batch_size = x[0].size(0)
    num_nodes = x[0].size(1)
    d_model = m.out_dim
    n_heads = m.n_heads
    
    # 主要计算来自:
    # 1. Q/K/V投影
    flops = 3 * num_nodes * d_model * d_model * batch_size
    # 2. 注意力计算
    flops += num_nodes * num_nodes * d_model * batch_size
    # 3. 信息聚合
    flops += num_nodes * d_model * d_model * batch_size
    
    m.total_ops = torch.DoubleTensor([flops])
def calculate_flops():
    model = actor_critic.ac
    
    # 包装forward方法以匹配thop要求
    class WrappedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, structure, inputs):
            val, _, _ = self.model(structure, inputs)
            return val  # 只返回主输出
    
    wrapped_model = WrappedModel(model).to(device)
    
    try:
        flops, params = profile(wrapped_model, 
                              inputs=(structure, inputs),
                              custom_ops={
                                  HGTConv: count_hgt_flops,  # 为自定义层添加计算规则
                                  GeneralConv: count_hgt_flops
                              })
        print(f"FLOPs: {flops / 1e9:.2f}G | Params: {params / 1e6:.2f}M")
    except Exception as e:
        print(f"计算失败: {str(e)}")

# calculate_flops()
# 参数量计算（保持不变）
params = sum(p.numel() for p in actor_critic.parameters() if p.requires_grad)
print(f"模型参数量: {params/1e6:.2f}M")