import torch
from ppo import utils
from ppo.evaluate_distill import evaluate
from ppo.metamorphmodel import ImitationNet
import os
import numpy as np
from ppo.arguments import get_args
from ppo.envs import make_vec_envs
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 加载训练好的模型
root_dir = os.path.dirname(os.path.abspath(__file__))
def load_model(model_path):
    model = ImitationNet()
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    return model

def get_actuator_indices(structure_matrix):
    """
    获取驱动器（3和4）的索引
    :param structure_matrix: 5x5的机器人结构矩阵
    :return: 驱动器索引列表
    """
    # 将矩阵展平
    flat_structure = structure_matrix.flatten()
    
    # 找到所有驱动器（3和4）的索引
    actuator_indices = [i for i, val in enumerate(flat_structure) if val in [3, 4]]
    
    return actuator_indices

# 快速评估函数
def quick_eval(model_path, num_evals=1):
    # 初始化参数
    env_name = "Walker-v0"  # 根据你的环境修改
    seed = 0
    num_processes = 1
    # eval_log_dir = "./eval_logs"

    args = get_args() 
    # 加载模型
    model = load_model(model_path)
    structures = []
    for i in range(10):
        save_path_structure = os.path.join(root_dir,"robot_universal/walker",str(i) + ".npz")
        np_data = np.load(save_path_structure)    #读取文件
        structure_data = []
        for key, value in np_data.items():  #将读取的原数据添加到新的预训练列表中#
            structure_data.append(value)
        structure_data = list(structure_data)
        # structure_data[0] = np.array([[4., 3., 1., 2., 2.],
        #                               [2., 4., 1., 2., 2.],
        #                               [2., 4., 3., 4., 4.],
        #                               [4., 3., 0., 0., 3.],
        #                               [2., 3., 0., 0., 1.]]
        #                             )            
        structures.append(Structure(*tuple(structure_data), i))     
    # 调用评估函数
    for structure in structures:   
        # 获取驱动器索引
        actuator_indices = get_actuator_indices(structure.body)
        print(f"结构 {structure.label} 的驱动器索引: {actuator_indices}")
        
        # 创建mask
        mask = torch.zeros(5*5, dtype=torch.bool)  # 5x5=25
        mask[actuator_indices] = True
        
        # 调整mask维度
        mask = mask.unsqueeze(0)  # 从[25]变为[1, 25]
        
        # 确保action和mask在相同设备上
        mask = mask.to(device)
        
        envs = make_vec_envs(env_name, (structure.body, structure.connections), args.seed, num_processes,
                        args.gamma, args.log_dir, device, False)
        obs_rms = utils.get_vec_normalize(envs).obs_rms        
        # avg_reward = evaluate(
        #     args.num_evals, model, obs_rms, env_name, (structure.body, structure.connections), 
        #     envs.action_space, args.seed, args.num_processes, args.log_dir, device, mask
        # )
        avg_reward = evaluate(
            args.num_evals, model, obs_rms, env_name, (structure.body, structure.connections), 
            envs.action_space, args.seed, args.num_processes, args.log_dir, device, mask
        )
        print(f"结构 {structure.label} 的平均奖励: {avg_reward:.2f}")

if __name__ == "__main__":
    # 修改为你的模型路径
    model_path = os.path.join(root_dir,"AAA_datasets","distilled_controllers","NSLC_Pretrain_walker_02_100","imitation_model.pt")
    quick_eval(model_path)