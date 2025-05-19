import os
import numpy as np
import shutil
import random
import math
import torch

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
sys.path.insert(2, curr_dir)

from ppo.NSLCrun import run_nslcppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from vec2morph import morph_to_vec
from ppo.metamorphmodel import ImitationNet
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure as BaseStructure
import torch
from ppo.evaluate import evaluate
from sklearn.cluster import KMeans
import collections
from ppo import utils
from ppo.envs import make_vec_envs
from ppo.arguments import get_args
from ppo.myPPOrun2 import run_ppo
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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
# 定义继承自BaseStructure的新类
class ExtendedStructure(BaseStructure):
    def __init__(self, body, connections, label):
        super().__init__(body, connections, label)
        self.pretrain_fitness = None  # 新增：预训练模型评估的fitness
        # self.ppo_fitness = 0.0       # 新增：PPO训练后的fitness

    def __str__(self):
        return f'\n\nStructure:\n{self.body}\nPretrain_F: {self.pretrain_fitness}\tID: {self.label}'

def evaluate_structure_with_pretrain(experiment_name,args, structure, controller, env_name, num_evals=1, num_processes=1):

    # 加载训练好的控制器
    actor_critic = controller
    # actor_critic.to(device)
    # actor_critic.eval()
    actuator_indices = get_actuator_indices(structure[0])
    
    # 创建mask
    mask = torch.zeros(5*5, dtype=torch.bool)  # 5x5=25
    mask[actuator_indices] = True
    
    # 调整mask维度
    mask = mask.unsqueeze(0)  # 从[25]变为[1, 25]
    
    # 确保action和mask在相同设备上
    mask = mask.to(device)    
    # 设置评估环境
    envs = make_vec_envs(args.env_name, structure, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    log_dir = args.log_dir

    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    utils.cleanup_log_dir(eval_log_dir)
    obs_rms = utils.get_vec_normalize(envs).obs_rms
    # 调用评估函数
    saving_convention = os.path.join(root_dir, "saved_data", experiment_name, "BFUCP")
    _ = run_ppo(structure, 250, saving_convention, actor_critic,args)  
    avg_reward = evaluate(
        num_evals=num_evals,
        actor_critic=actor_critic,
        obs_rms=obs_rms,
        env_name=env_name,
        robot_structure=structure,
        action_space=None,  # 从环境中获取
        seed=0,
        num_processes=num_processes,
        eval_log_dir=eval_log_dir,
        device=device,
        mask=mask
    )
    return avg_reward

def run_se(experiment_name, load_name, structure_shape, pop_size, max_evaluations, train_iters, num_cores):
    print()
    pretrain_model_path = os.path.join(root_dir, "AAA_datasets", "distilled_controllers", load_name, "imitation_model.pt")
    imitation_model = ImitationNet()
    imitation_model.load_state_dict(torch.load(pretrain_model_path))
    imitation_model.to(device)
    imitation_model.eval()
    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    start_gen = 0

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)
    
    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        print("Override? (y/n/c): ", end="")
        ans = input()
        if ans.lower() == "y":
            shutil.rmtree(home_path)
            print()
        elif ans.lower() == "c":
            print("Enter gen to start training on (0-indexed): ", end="")
            start_gen = int(input())
            is_continuing = True
            print()
        else:
            return

    ### STORE META-DATA ##
    if not is_continuing:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        
        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.close()

    else:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:
            if count == 0:
                pop_size = int(line.split()[1])
            if count == 1:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 2:
                max_evaluations = int(line.split()[1])
            if count == 3:
                train_iters = int(line.split()[1])
                tc.change_target(train_iters)
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' + 
            f'max evals: {max_evaluations}, train iters {train_iters}.')
        
        f.close()

    ### GENERATE // GET INITIAL POPULATION ###
    structures = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = 0
    
    #generate a population
    if not is_continuing: 
        for i in range (pop_size):
            
            temp_structure = sample_robot(structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(structure_shape)

            structures.append(ExtendedStructure(*temp_structure, i))
            population_structure_hashes[hashable(temp_structure[0])] = True
            num_evaluations += 1

    #read status from file
    else:
        for g in range(start_gen+1):
            for i in range(pop_size):
                save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(g), "structure", str(i) + ".npz")
                np_data = np.load(save_path_structure)
                structure_data = []
                for key, value in np_data.items():
                    structure_data.append(value)
                structure_data = tuple(structure_data)
                population_structure_hashes[hashable(structure_data[0])] = True
                # only a current structure if last gen
                if g == start_gen:
                    structures.append(ExtendedStructure(*structure_data, i))
        num_evaluations = len(list(population_structure_hashes.keys()))
        generation = start_gen


    while True:

        ### UPDATE NUM SURVIORS ###			
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))


        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller")
        valid_num_path = os.path.join(root_dir, "saved_data", experiment_name, "valid_rate.txt")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass
        

        ### SAVE POPULATION DATA ###
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)
        args = get_args()
        ### TRAIN GENERATION

        #better parallel
        group = mp.Group()   
        #if generation==0 or generation==start_gen: #  
        for structure in structures:
            structure.pretrain_fitness = evaluate_structure_with_pretrain(experiment_name,args, (structure.body, structure.connections), imitation_model, args.env_name, args.num_evals, args.num_processes)
            imitation_model.load_state_dict(torch.load(pretrain_model_path))
            if structure.is_survivor:
                save_path_controller_part = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                save_path_controller_part_old = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation-1), "controller",
                    "robot_" + str(structure.prev_gen_label) + "_controller" + ".pt")
                
                print(f'Skipping training for {save_path_controller_part}.\n')
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                except:
                    print(f'Error coppying controller for {save_path_controller_part}.\n')
            else:        
                ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label))
                group.add_job(run_nslcppo, ppo_args, callback=structure.set_reward)

        group.run_jobs(num_cores)
     

        #not parallel
        #for structure in structures:
        #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()
        
        structures = sorted(structures, key=lambda structure: structure.pretrain_fitness, reverse=True)
        if generation % 5 == 0:
            structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        #SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.pretrain_fitness) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()

         ### CHECK EARLY TERMINATION ###
        if num_evaluations >= max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            # 新增：保存最终结果
            final_result_path = os.path.join(root_dir, "saved_data", experiment_name, "final_results")
            os.makedirs(final_result_path, exist_ok=True)
            ALL_structure_hashes = {}
            # 使用population_structure_hashes来获取所有独特形态
            unique_structures = []
            for g in range(generation + 1):
                gen_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(g))
                if not os.path.exists(gen_path):
                    continue
                    
                # 读取每代的结构
                structure_dir = os.path.join(gen_path, "structure")
                for structure_file in os.listdir(structure_dir):
                    if structure_file.endswith('.npz'):
                        structure_path = os.path.join(structure_dir, structure_file)
                        np_data = np.load(structure_path)
                        structure_data = tuple([np_data[key] for key in np_data])
                        structure_hash = hashable(structure_data[0])
                        
                        # 如果哈希值在population_structure_hashes中，则记录
                        if structure_hash not in ALL_structure_hashes:
                            # 获取对应的fitness
                            output_file = os.path.join(gen_path, "output.txt")
                            with open(output_file, 'r') as f:
                                for line in f:
                                    label, pretrain_fitness, fitness = line.strip().split('\t\t')
                                    if label == structure_file.split('.')[0]:
                                        unique_structures.append((g, int(label), float(pretrain_fitness), float(fitness), structure_data))
                                        ALL_structure_hashes[structure_hash] = True
                                        break
            
            # 保存所有独特形态
            for g, label, pretrain_fitness, fitness, structure_data in unique_structures:
                new_name = f"{g}_{label}.npz"
                np.savez(os.path.join(final_result_path, new_name), *structure_data)
            sorted_by_generation = sorted(unique_structures, key=lambda x: (x[0], x[1]))
            generation_sequence_path = os.path.join(final_result_path, "generation_sequence.txt")
            
            with open(generation_sequence_path, 'w') as f:
                f.write("Generation\tLabel\tPretrain_Fitness\tPPO_Fitness\n")
                for gen, label, pretrain_fitness, fitness, _ in sorted_by_generation:
                    f.write(f"{gen}\t{label}\t{pretrain_fitness}\t{fitness}\n")
            
            print(f"Generation sequence saved to {generation_sequence_path}")
            
            # 计算并保存最佳和平均fitness
            all_fitness = [f for _, _, _, f, _ in unique_structures]
            best_fitness = max(all_fitness)
            avg_fitness = sum(all_fitness) / len(all_fitness)
            
            # 新增：按生成数量分段统计
            segment_sizes = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250,275,300,325,350,375,400,425,450,475,500,525,550,575,600,625,650,675,700,725,750,775,800,825,850,875,900,925,950,975,1000]  # 你可以根据需要调整这些分段大小
            segment_stats = []
            
            # 按生成顺序排序（通过代数和label）
            sorted_by_generation = sorted(unique_structures, key=lambda x: (x[0], x[1]))
            
            # 计算每个分段的统计信息
            for size in segment_sizes:
                if len(sorted_by_generation) >= size:
                    # 获取当前分段
                    segment = sorted_by_generation[:size]
                    segment_fitness = [s[3] for s in segment]
                    segment_best = max(segment_fitness)
                    segment_avg = sum(segment_fitness) / len(segment_fitness)
                    
                    # 记录统计信息
                    start_gen = segment[0][0]
                    end_gen = segment[-1][0]
                    segment_stats.append((size, start_gen, end_gen, segment_best, segment_avg))
            
            # 保存所有统计信息
            with open(os.path.join(final_result_path, "summary.txt"), 'w') as f:
                f.write(f"Best Fitness: {best_fitness}\n")
                f.write(f"Average Fitness: {avg_fitness}\n")
                f.write(f"Total Unique Structures: {len(unique_structures)}\n")
                
                # 写入分段统计
                f.write("\nGeneration Segment Statistics by Quantity:\n")
                for size, start_gen, end_gen, seg_best, seg_avg in segment_stats:
                    f.write(f"First {size} structures (Generations {start_gen}-{end_gen}):\n")
                    f.write(f"  Best Fitness: {seg_best}\n")
                    f.write(f"  Average Fitness: {seg_avg}\n")
            
            print(f"Final results saved to {final_result_path}")
            print(f"Total unique structures generated: {len(unique_structures)}")
            return

        print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)} percent of DESIGNS:\n')
        print(structures[:num_survivors])
        
        
        # K-Means Clustering
        bodies = []
        for structure in structures:
            bodies.append(structure.body)
        bodies_vec = morph_to_vec(torch.tensor(np.array(bodies)))
        clusterer = KMeans(n_clusters=3, random_state=9)
        y_pred = clusterer.fit_predict(bodies_vec)
        cen = clusterer.cluster_centers_
        structures_k = collections.defaultdict(list)
        for ii,kk in enumerate(y_pred):
            structures_k[kk].append(structures[ii])
        
        # keep survivors
        ## the best one in the whole pop
        survivors = []
        survivors.append(structures[0])
        ## the best one in each class (larger than n_elite=3)
        n_elite = 5
        for kk in range(3):
            if (y_pred==kk).sum()>n_elite:
                survivors.append(structures_k[kk][0])
        
        for i in range(len(survivors)):
            survivors[i].is_survivor = True
            survivors[i].prev_gen_label = survivors[i].label
            survivors[i].label = i
        
        # rank representatives and compute portions
        rep_scores = []
        for kk in range(3):
            bodies_kk = bodies_vec[list(np.where(y_pred==kk)[0]),:]
            dists = ((bodies_kk - cen[kk,:])**2).sum(dim=1)
            index = torch.argmin(dists)
            rep_scores.append(structures_k[kk][index].fitness)
        ranks = torch.sort(torch.tensor(rep_scores))[1] + 1
        alpha = 0.75
        portions = ((pop_size - len(survivors))*(alpha**ranks/(alpha**ranks).sum())).round()
        
        # mutate
        total_attempts = 0
        num_children = 0
        num_survivors = len(survivors)
        for kk in range(3):
            portion = portions[kk]
            c = 0
            while c < portion and num_evaluations < max_evaluations:
                parent_index = c % len(structures_k[kk])
                # child = None
                # while child is None or (isinstance(child, np.ndarray) and np.all(child == None)):
                child, attempts = mutate(structures_k[kk][parent_index].body.copy(), mutation_rate = 0.1, num_attempts=50)
                total_attempts += attempts
                
                if child != None and hashable(child[0]) not in population_structure_hashes:
                    survivors.append(ExtendedStructure(*child, num_survivors + num_children))
                    population_structure_hashes[hashable(child[0])] = True
                    num_children += 1
                    num_evaluations += 1
                    c += 1
        valid_rate = (pop_size - num_survivors)/total_attempts
        
        '''
        ### CROSSOVER AND MUTATION ###
        # save the survivors
        survivors = structures[:num_survivors]

        #store survivior information to prevent retraining robots
        for i in range(num_survivors):
            structures[i].is_survivor = True
            structures[i].prev_gen_label = structures[i].label
            structures[i].label = i

        # for randomly selected survivors, produce children (w mutations)
        
        total_attempts = 0
        
        num_children = 0
        while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:

            parent_index = random.sample(range(num_survivors), 1)
            child, attempts = mutate(survivors[parent_index[0]].body.copy(), mutation_rate = 0.1, num_attempts=50)
            
            total_attempts += attempts

            if child != None and hashable(child[0]) not in population_structure_hashes:
                
                # overwrite structures array w new child
                structures[num_survivors + num_children] = Structure(*child, num_survivors + num_children)
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1
        
        valid_rate = (pop_size - num_survivors)/total_attempts
        '''
        f = open(valid_num_path, "a")

        out = ""
        out += str(generation) + "\t\t" + str(total_attempts) + "\t\t" + str(pop_size - num_survivors) + "\t\t" + str(valid_rate) + "\n"
        f.write(out)

        f.close()
        
        structures = survivors
        #structures = structures[:num_children+num_survivors]

        generation += 1