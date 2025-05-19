import os
import torch
import numpy as np
import shutil
import random
import math
from ppo.evaluate import evaluate
from ppo import utils
import sys     #定义训练的路径#
curr_dir = os.path.dirname(os.path.abspath(__file__))  #打印当前文件的绝对路径,获取当前文件上一层目录#
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)   #定义搜索路径的优先级顺序#
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
import datetime
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, mutate_new,mutate_new_nozero,TerminationCondition, Structure
from ppo.arguments import get_args
from ppo.envs import make_vec_envs
import copy
import collections
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def inverted_ga(structures, actor_critic, args, max_evaluations, pop_size, experiment_name, num_evaluations, population_structure_hashes,generation):
    # for i in range(len(structures)):
    #     structures[i].label = str(generation) +"_"+str(i)
    structures_k = collections.defaultdict(list)
    survivors = []
    for i in range(len(structures)):
        structures_k[i].append(structures[i])
    for i in range(len(structures_k)):  
        # percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        # num_survivors = max(2, math.ceil(pop_size * percent_survival))        
        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "structure")   #拼接创建此代幸存者的结构和控制器的保存路径#
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass
        structures_k[i][0].is_survivor = True
        structures_k[i][0].prev_gen_label = structures_k[i][0].label
        num_children = 0
        
        while num_children < 3 and num_evaluations < max_evaluations:

            child = mutate_new_nozero(structures_k[i][0].body.copy(), mutation_rate = 0.1, edge_mutation_rate=0.12, num_attempts=100)
            if child != None and hashable(child[0]) not in population_structure_hashes:
                structures_k[i].append(Structure(*child, str(generation)+"_"+str(i)+"_"+str(1 + num_children)))
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                

        for j in range (len(structures_k[i])):
            temp_path = os.path.join(save_path_structure, str(structures_k[i][j].label))
            np.savez(temp_path, structures_k[i][j].body, structures_k[i][j].connections)   
        for structure in structures_k[i]:
                #structure.compute_fitness()
                structure_bc = (structure.body, structure.connections)
                envs = make_vec_envs(args.env_name, structure_bc, args.seed, args.num_processes,
                            args.gamma, args.log_dir, device, False)
                obs_rms = utils.get_vec_normalize(envs).obs_rms
                log_dir = args.log_dir
                if save_path_controller != None:
                    log_dir = os.path.join(save_path_controller, log_dir, "robot_" + str(structure.label))
                eval_log_dir = log_dir + "_eval"
                utils.cleanup_log_dir(log_dir)
                utils.cleanup_log_dir(eval_log_dir)
                fitness = 0
                for k in range(3):
                    fitness += -evaluate(args.num_evals, actor_critic, obs_rms, args.env_name, structure_bc, envs.action_space,args.seed,
                        args.num_processes, eval_log_dir, device)
                structure.fitness = fitness/3 
                
        structures_k[i] = sorted(structures_k[i], key=lambda structure: structure.fitness, reverse=True)   #按照structure列表的structure.fitness属性降序排列#
        num_evaluations += 1

        # for structure in structures_k[i]:
        #     if structure.is_survivor != True:
        #         a = copy.deepcopy(structure)
        #         structures_all.append(a)
        #SAVE RANKING TO FILE
        #temp_path = os.path.join(root_dir, "saved_data", experiment_name, "output.txt")
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "a")

        out = ""
        for structure in structures_k[i]:
            out += str(structure.label) + "\t\t" + str(-structure.fitness) + "\n"
        f.write(out)
        f.close()

        survivors.append(structures_k[i][0]) #RUCP,择劣选取
        # survivors.append(structures_k[i][-1]) #OU,择优选取
        # survivors.append(random.choice(structures_k[i])) #RU,随机选取
        ### CHECK EARLY TERMINATION ###
        if num_evaluations == max_evaluations:
            return survivors, num_evaluations, population_structure_hashes, generation
        
    generation += 1
    return survivors, num_evaluations, population_structure_hashes, generation



