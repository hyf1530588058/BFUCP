import os
import random
import _pickle as pickle
import math
import numpy as np
from skimage.measure import label

import shutil
import sys
from collections import defaultdict
from scipy.spatial.distance import cdist
from ppo.MLP_model import Policy
import torch
from ppo.envs import make_vec_envs
import glob
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
# nslc_dir = os.path.join(root_dir, 'nslc')
# os.makedirs(nslc_dir, exist_ok=True)
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
from ppo.NSLCrun import run_nslcppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure
from ppo.run import get_args

args = get_args()
device = torch.device("cuda:0" if args.cuda else "cpu")

class ArchiveItem(Structure):
    def __init__(self, body, connections, label=None, novelty=0, curiosity=0):
        super().__init__(body, connections, label)
        self.novelty = novelty
        self.curiosity = curiosity
        self._controller = None  # 使用私有属性存储
        self.controller_path = None  # 新增：存储控制器路径

    @property
    def controller(self):
        """获取控制器"""
        if self._controller is None and self.controller_path is not None:
            # 延迟加载控制器
            try:
                self._controller = self.load_controller()
            except Exception as e:
                print(f'Error loading controller: {str(e)}')
                return None
        return self._controller

    @controller.setter
    def controller(self, value):
        """设置控制器"""
        if not isinstance(value, torch.nn.Module):
            raise ValueError("Controller must be a PyTorch Module")
        self._controller = value

    def save_controller(self, path):
        """保存控制器到文件"""
        if self._controller is None:
            raise ValueError("No controller to save")
        torch.save(self._controller.state_dict(), path)
        self.controller_path = path

    def load_controller(self):
        """从文件加载控制器"""
        if self.controller_path is None:
            raise ValueError("No controller path set")
        if not os.path.exists(self.controller_path):
            raise FileNotFoundError(f"Controller file not found: {self.controller_path}")
        
        envs = make_vec_envs(args.env_name, (self.body, self.connections), args.seed, args.num_processes,
                            args.gamma, args.log_dir, device, False)
        controller = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kwargs={'recurrent': args.recurrent_policy})
        controller.load_state_dict(torch.load(self.controller_path))
        return controller

    def __str__(self):
        return f'\n\nArchiveItem:\n{self.body}\nF: {self.fitness}\tN: {self.novelty}\tC: {self.curiosity}\tID: {self.label}'

def type_composition(matrix):
    types, counts = np.unique(matrix, return_counts=True)
    comp_dict = {t: counts[i]/25 for i, t in enumerate(types)}
    # 补全缺失类型为0
    return [comp_dict.get(i, 0.0) for i in range(1,5)]  # 排除空体素（0）

def spatial_features(matrix):
    # 区域密度
    quadrants = [
        matrix[:2, :2], matrix[:2, 3:], 
        matrix[3:, :2], matrix[3:, 3:]
    ]
    densities = [np.count_nonzero(q)/4 for q in quadrants]
    
    # 对称性（水平、垂直）
    horizontal_sym = np.mean(matrix == np.flip(matrix, axis=0))
    vertical_sym = np.mean(matrix == np.flip(matrix, axis=1))

    return densities + [horizontal_sym, vertical_sym]   

def actuator_features(matrix):
    # 获取所有驱动器的坐标
    h_actuators = np.argwhere(matrix == 3)
    v_actuators = np.argwhere(matrix == 4)
    
    # 计算水平驱动器周围硬体素（1）比例
    h_hard_ratio = []
    for y, x in h_actuators:
        neighbors = matrix[max(0,y-1):y+2, max(0,x-1):x+2]
        h_hard_ratio.append(np.sum(neighbors == 1) / (neighbors.size - 1))
        
    v_soft_ratio = []
    for y, x in v_actuators:
        neighbors = matrix[max(0,y-1):y+2, max(0,x-1):x+2]
        v_soft_ratio.append(np.sum(neighbors == 2) / (neighbors.size - 1))
        
    return [
        np.mean(h_hard_ratio) if h_actuators.size >0 else 0,
        np.mean(v_soft_ratio) if v_actuators.size >0 else 0,
        len(h_actuators)/25,  # 水平驱动器密度
        len(v_actuators)/25    # 竖直驱动器密度
    ]

def connectivity(matrix):
    """计算机器人结构的连通性"""
    # 硬体素连通性
    hard_mask = (matrix == 1).astype(int)
    labeled = label(hard_mask)
    counts = np.bincount(labeled.flat)[1:]
    
    if len(counts) > 0:
        hard_conn = np.max(counts)
    else:
        hard_conn = 0
    
    # 软体素连通性
    soft_mask = (matrix == 2).astype(int)
    labeled = label(soft_mask)
    counts = np.bincount(labeled.flat)[1:]
    
    if len(counts) > 0:
        soft_conn = np.max(counts)
    else:
        soft_conn = 0
    
    return [hard_conn/25, soft_conn/25]

def calculate_archive_threshold(archive):
    """计算存档中各形态之间距离的平均值作为加入阈值"""
    if len(archive) < 2:
        return 0.0
    
    distances = []
    for i in range(len(archive)):
        for j in range(i+1, len(archive)):
            dist = calculate_distance(archive[i], archive[j])
            distances.append(dist)
    
    return np.mean(distances) if distances else 0.0

def run_nslc(experiment_name, structure_shape, pop_size, max_evaluations, train_iters, num_cores, archive_size,
            k=5, epsilon=0.3, reward=1, penalty=0.5):
    ### 初始化设置 ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    archive = []  # 存档容器
    num_evaluations = 0
    generation = 0
    population_structure_hashes = {}
    tc = TerminationCondition(train_iters)
    # 创建或清空实验目录
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Override? (y/n): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        else:
            return

    ### 初始化存档 ###
    if len(archive) == 0:
        print("Initializing archive...")
        group = mp.Group()  # 新增：用于并行评估
        while len(archive) < pop_size:
            body, connections = sample_robot(structure_shape)
            while hashable(body) in population_structure_hashes:
                body, connections = sample_robot(structure_shape)

            population_structure_hashes[hashable(body)] = True
            new_item = ArchiveItem(body, connections, label=len(archive))
            archive.append(new_item)
            num_evaluations += 1

            save_path_controller = os.path.join(home_path, "generation_0", 
                                             "controller")
            os.makedirs(os.path.dirname(save_path_controller), exist_ok=True)
            ppo_args = ((new_item.body, new_item.connections), tc, 
                       (save_path_controller, new_item.label))
            group.add_job(run_nslcppo, ppo_args, callback=new_item.set_reward)
        
        # 新增：运行所有初始评估
        group.run_jobs(num_cores)
        for i, offspring in enumerate(archive):
            # 计算适应度
            offspring.compute_fitness()
            
        update_archive_novelty(archive, k)
        save_generation_data(home_path, generation, archive, num_evaluations, max_evaluations)
        generation += 1

    #全局标签计数器
    global_label_counter = len(archive)  # 初始化为当前存档大小

    ### NSLC 主循环 ###
    while num_evaluations < max_evaluations:
        print(f'\nGeneration {generation}, Evaluations: {num_evaluations}/{max_evaluations}')
        
        # 1. 选择父代
        archive_sorted = sorted(archive, key=lambda x: x.curiosity, reverse=True)
        structure_parents = archive_sorted[:pop_size]
        
        # 2. 生成子代
        structure_offspring = []
        for i, parent in enumerate(structure_parents):
            while True:
                child = mutate(parent.body.copy(), mutation_rate = 0.1, num_attempts=50)
                if child != None and hashable(child[0]) not in population_structure_hashes:
                    break
            # 使用全局标签计数器
            structure_offspring.append(ArchiveItem(child[0], child[1], label=global_label_counter))
            global_label_counter += 1  # 递增计数器
            population_structure_hashes[hashable(child[0])] = True
        
        # 3. 评估子代
        group = mp.Group()
        for offspring in structure_offspring:
            save_path_controller = os.path.join(home_path, "generation_" + str(generation), 
                                             "controller")
            try:
                os.makedirs(save_path_controller)
            except:
                pass
            ppo_args = ((offspring.body, offspring.connections), tc, (save_path_controller, offspring.label))
            group.add_job(run_nslcppo, ppo_args, callback=offspring.set_reward)
        group.run_jobs(num_cores)
        
        # 4. 计算子代属性并更新存档
        threshold = calculate_archive_threshold(archive)
        for i, offspring in enumerate(structure_offspring):
            # 计算适应度
            offspring.compute_fitness()
            # 计算新颖度
            distances = []
            for item in archive:
                dist = calculate_distance(offspring, item)
                distances.append(dist)
            k_nearest = sorted(distances)[:k]
            offspring.novelty = np.mean(k_nearest)

            if len(archive) < archive_size:
                # 存档未满，直接加入
                archive.append(offspring)
                structure_parents[i].curiosity += reward
            else:            
                # 检查是否加入存档
                distances = [calculate_distance(offspring, item) for item in archive]
                nearest_item = archive[np.argmin(distances)]
            
                if np.min(distances) > threshold:
                    # 直接加入存档
                    archive.remove(nearest_item)
                    structure_parents[i].curiosity += reward
                else:
                    # ε-支配准则
                    if (offspring.novelty >= (1 - epsilon) * nearest_item.novelty and
                        offspring.fitness >= (1 - epsilon) * nearest_item.fitness and
                        (offspring.novelty - nearest_item.novelty) * nearest_item.fitness > 
                        -(offspring.fitness - nearest_item.fitness) * nearest_item.novelty):
                        archive.remove(nearest_item)
                        structure_parents[i].curiosity += reward
                    else:
                        structure_parents[i].curiosity = structure_parents[i].curiosity - penalty
            
            num_evaluations += 1
        
        # 5. 更新存档中所有个体的新颖度
        update_archive_novelty(archive, k)
        
        # 6. 保存当前代数据
        save_generation_data(home_path, generation, archive, num_evaluations, max_evaluations)
        generation += 1

    print("\nOptimization complete!")

def calculate_distance(item1, item2):
    """计算两个个体之间的距离"""
    type_dist = np.linalg.norm(np.array(type_composition(item1.body)) - 
                np.array(type_composition(item2.body)))
    spatial_dist = np.linalg.norm(np.array(spatial_features(item1.body)) - 
                   np.array(spatial_features(item2.body)))
    actuator_dist = np.linalg.norm(np.array(actuator_features(item1.body)) - 
                    np.array(actuator_features(item2.body)))
    conn_dist = np.linalg.norm(np.array(connectivity(item1.body)) - 
                np.array(connectivity(item2.body)))
    
    return 0.3*type_dist + 0.4*spatial_dist + 0.2*actuator_dist + 0.1*conn_dist

def update_archive_novelty(archive, k):
    """更新存档中所有个体的新颖度"""
    for i, item in enumerate(archive):
        distances = []
        for j, other in enumerate(archive):
            if i == j:
                continue
            distances.append(calculate_distance(item, other))
        k_nearest = sorted(distances)[:k]
        item.novelty = np.mean(k_nearest)

def copy_controllers_to_archive(gen_path, archive, home_path):
    """将archive中所有个体的控制器文件复制到archive_controller文件夹中"""
    controller_path = os.path.join(gen_path, "archive_controller")
    os.makedirs(controller_path, exist_ok=True)
    
    for item in archive:
        # 构造控制器文件的原始路径
        original_path = os.path.join(home_path, f"generation_*/controller/robot_{item.label}_controller.pt")
        matches = glob.glob(original_path)
        
        if matches:
            # 找到对应的控制器文件
            src_path = matches[0]
            dest_path = os.path.join(controller_path, f"robot_{item.label}_controller.pt")
            
            # 复制文件
            shutil.copy(src_path, dest_path)
            print(f'Copied controller for item {item.label} from {src_path} to {dest_path}')
        else:
            print(f'Warning: No controller found for item {item.label}')

def save_generation_data(home_path, generation, archive, num_evaluations, max_evaluations):
    """保存当前代数据"""
    gen_path = os.path.join(home_path, f"generation_{generation}")
    os.makedirs(gen_path, exist_ok=True)
    
    # 保存结构
    structure_path = os.path.join(gen_path, "structure")
    os.makedirs(structure_path, exist_ok=True)
    
    # 保存控制器（仅在第一代、每隔10代和最后一代）
    if generation == 0 or generation % 1 == 0 or num_evaluations >= max_evaluations:
        copy_controllers_to_archive(gen_path, archive, home_path)
    
    for i, item in enumerate(archive):
        # 保存结构数据
        np.savez(os.path.join(structure_path, f"{item.label}.npz"), 
                 item.body, item.connections)
    
    # 保存排名
    ranking_path = os.path.join(gen_path, "output.txt")
    with open(ranking_path, 'w') as f:
        for item in sorted(archive, key=lambda x: x.fitness, reverse=True):
            f.write(f"{item.label}\t\t{item.fitness}\t\t{item.novelty}\t\t{item.curiosity}\n")