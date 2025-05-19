import numpy as np
import os
import sys
import json
import argparse
import random
import pickle
from multiprocessing import Pool
from functools import partial
import torch.multiprocessing as mp
curr_dir = os.path.dirname(os.path.abspath(__file__))
external_dir = os.path.join(curr_dir, 'externals')
sys.path.insert(0, curr_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
import time
from collections import deque
import torch
from ppo import utils
from ppo.arguments import get_args
from ppo.evaluate import evaluate
from ppo.envs import make_vec_envs
from utils.algo_utils import Structure
from a2c_ppo_acktr import algo          
from a2c_ppo_acktr.algo import gail
from ppo.MLP_model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
import evogym.envs
import itertools
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import datetime
def load_individuals(args, experiment_name, generation, num_individuals=None):
    """
    从指定实验文件夹加载所有个体及其控制器
    :param num_individuals: 要加载的个体数量，如果为None则加载全部
    :return: 按output.txt顺序排序的个体列表
    """
    base_path = os.path.join(curr_dir, "saved_data", experiment_name, f"generation_{generation}")
    individuals = []
    
    # 1. 读取output.txt文件
    output_path = os.path.join(base_path, "output.txt")
    if not os.path.exists(output_path):
        print(f"Error: output.txt not found in {base_path}")
        return individuals
        
    # 读取并解析output.txt
    with open(output_path, 'r') as f:
        lines = f.readlines()
    
    # 解析每行数据，格式为：标签 适应度 其他数据...
    robot_order = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 2:
            robot_order.append(int(parts[0]))  # 第一列是机器人标签
    
    # 2. 如果指定了加载数量，截取前num_individuals个
    if num_individuals is not None:
        robot_order = robot_order[:num_individuals]
    
    # 3. 按顺序加载个体
    structure_dir = os.path.join(base_path, "structure")
    controller_dir = os.path.join(base_path, "archive_controller")
    
    for robot_id in robot_order:
        print(robot_id)
        structure_file = f"{robot_id}.npz"
        if not os.path.exists(os.path.join(structure_dir, structure_file)):
            print(f"Warning: Structure file for robot {robot_id} not found, skipping...")
            continue
            
        try:
            # 加载机器人结构
            structure_path = os.path.join(structure_dir, structure_file)
            data = np.load(structure_path)
            structure_data = []
            for key, value in itertools.islice(data.items(), 2):
                structure_data.append(value)
            structure_data = list(structure_data)        
            structure = Structure(*tuple(structure_data), str(robot_id))
            
            # 加载控制器
            controller_path = os.path.join(controller_dir, f"robot_{robot_id}_controller.pt")
            if not os.path.exists(controller_path):
                print(f"Warning: Controller for robot {robot_id} not found, skipping...")
                continue

            envs = make_vec_envs(args.env_name, (structure.body, structure.connections), args.seed, 1,
                                args.gamma, args.log_dir, device, False)
                        
            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy})
            actor_critic.to(device) 
            actor_critic.load_state_dict(torch.load(controller_path)[0].state_dict())
            
            individuals.append({
                'id': str(robot_id),
                'structure': structure,
                'controller': actor_critic
            })
            
        except Exception as e:
            print(f"Error loading robot {robot_id}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(individuals)} individuals in order from output.txt")
    return individuals

def get_driver_mask(structure_body):
    """
    生成驱动器掩码：驱动器位置为1，非驱动器位置为0
    """
    return ((structure_body == 3) | (structure_body == 4)).flatten()

def pad_action(action, structure_body):
    """
    根据机器人结构填充动作向量
    - 驱动器位置：保留原始动作值
    - 非驱动器位置：填充0
    """
    # 生成驱动器掩码
    driver_mask = get_driver_mask(structure_body)
    
    # 初始化25维动作向量
    padded_action = torch.zeros((action.shape[0], 25), device=device)
    
    # 将原始动作值填充到驱动器位置
    padded_action[:, driver_mask] = action
    
    return padded_action

def simulate_parallel(individual, args):
    """
    单个个体的仿真函数，用于并行执行
    """
    try:
        # 初始化存储列表
        obs_list = []
        act_list = []
        mask_list = []
        structure_list = []
        
        # 正确获取控制器
        controller = individual['controller'].to(device)
        
        # 获取机器人结构
        structure_body = individual['structure'].body
        mask = ((structure_body == 3) | (structure_body == 4)).flatten()
        
        # 评估次数
        num_evals = 100  # 与evaluate.py一致
        
        for eval_idx in range(num_evals):  # 每个个体运行num_evals次
            # 创建环境
            env = make_vec_envs(
                args.env_name, 
                (individual['structure'].body, individual['structure'].connections), 
                args.seed+eval_idx,  # 使用训练时的seed
                1,  # num_processes=1
                args.gamma,  # 与训练时一致
                args.log_dir, 
                device, 
                False  # 不允许渲染
            )
            
            # 初始化隐藏状态和掩码
            hidden_state = torch.zeros(1, controller.recurrent_hidden_state_size, device=device)
            masks = torch.zeros(1, 1, device=device)
            eval_rewards = []
            # 初始化存储
            run_obs = []
            run_act = []
            
            # 重置环境
            obs = env.reset()
            run_obs.append(obs)
            
            # 仿真循环
            for t in range(500):  # 每次运行最多500步
                with torch.no_grad():
                    value, action, _, hidden_state = controller.act(
                        obs,
                        hidden_state,
                        masks,
                        deterministic=True)  # 使用确定性策略
                
                # 根据机器人结构填充动作向量
                padded_action = pad_action(action, structure_body)
                
                # 存储动作
                run_act.append(padded_action)
                
                # 执行环境步进
                obs, _, done, infos = env.step(action)
                run_obs.append(obs)
                
                # 更新掩码
                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=device)
                
                if done:
                    break

            for info in infos:
                if 'episode' in info.keys():
                    eval_rewards.append(info['episode']['r'])     

            # print(individual['id'],np.mean(eval_rewards))

            if len(run_obs) != len(run_act) + 1:
                print(f"Warning: Mismatch in obs/act length: {len(run_obs)} vs {len(run_act)}")
                continue
                
            
            # 将本次运行的观测、动作、掩码和结构添加到总列表
            obs_list.append(torch.stack(run_obs[:-1]))
            act_list.append(torch.stack(run_act))
            mask_list.append(torch.tensor(mask).unsqueeze(0).repeat(len(run_act), 1))
            structure_list.append(torch.tensor(structure_body).unsqueeze(0).repeat(len(run_act), 1, 1))
            
            # 关闭环境
            env.close()
        
        # 将所有数据一次性转移到CPU并转换为numpy
        obs_np = torch.cat(obs_list).cpu().numpy()
        act_np = torch.cat(act_list).cpu().numpy()
        mask_np = torch.cat(mask_list).cpu().numpy()
        structure_np = torch.cat(structure_list).cpu().numpy()
        
        # 释放CUDA内存
        del controller, env, obs, action
        torch.cuda.empty_cache()
        
        return {
            'obs': obs_np,
            'act': act_np,
            'mask': mask_np,
            'structure': structure_np
        }
        
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        return None


def create_dataset(args, num_cores, experiment_name, generation, num_individuals=None):
    """
    主函数：创建数据集
    """
    seed = 0 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 1. 设置正确的启动方法
    mp.set_start_method('spawn', force=True)
    
    # 2. 加载所有个体
    individuals = load_individuals(args, experiment_name, generation, num_individuals)
    
    if not individuals:
        print("Error: No individuals loaded, cannot create dataset")
        return
    
    # 3. 创建保存目录
    save_dir = os.path.join(curr_dir, "AAA_datasets")
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. 并行仿真
    with mp.Pool(processes=num_cores) as pool:
        results = pool.map(partial(simulate_parallel, args=args), individuals)
    
    # 5. 合并数据
    obs = []
    act = []
    mask = []
    structure = []
    for dataset in results:
        if dataset is None:
            continue
        if len(dataset['obs']) != len(dataset['act']):
            print(f"Warning: Skipping invalid dataset with mismatched lengths: "
                  f"{len(dataset['obs'])} vs {len(dataset['act'])}")
            continue
        obs.extend(dataset['obs'])
        act.extend(dataset['act'])
        mask.extend(dataset['mask'])
        structure.extend(dataset['structure'])
    # 检查是否有有效数据
    if not obs or not act:
        print("Error: No valid data collected")
        return None
    
    # 6. 数据分割与保存
    obs = np.array(obs)
    act = np.array(act)
    mask = np.array(mask)
    structure = np.array(structure)
    idx = np.arange(obs.shape[0])
    np.random.shuffle(idx)
    split = int(0.8 * obs.shape[0])
    
    to_save = {
        'obs_train': obs[idx[:split]],
        'act_train': act[idx[:split]],
        'mask_train': mask[idx[:split]],
        'structure_train': structure[idx[:split]],
        'obs_val': obs[idx[split:]],
        'act_val': act[idx[split:]],
        'mask_val': mask[idx[split:]],
        'structure_val': structure[idx[split:]],
    }
    v1 = "_001_10_100"
    path = os.path.join(save_dir, f"{experiment_name}{v1}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(to_save, f, protocol=-1)

def main(args, experiment_name, num_cores):
    create_dataset(args, num_cores, experiment_name, generation=2, num_individuals=6)

if __name__ == "__main__":
    print('run_universal start at ', datetime.datetime.now())
    args = get_args()
    experiment_name = "NSLC_test"
    num_cores = 6
    main(args, experiment_name, num_cores)
    print('run_universal over at ', datetime.datetime.now())

# python create_dataset.py --env-name "Walker-v0" 2>&1 | tee -a create_dataset_test.log
