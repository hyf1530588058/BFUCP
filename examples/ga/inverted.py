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
from ppo.mygaPPOrun import run_gappo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure
from ppo.mygaPPOmodel import Policy
from ppo.arguments import get_args
from ppo.envs import make_vec_envs
import copy
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def inverted_ga(structures, actor_critic, args, max_evaluations, pop_size, experiment_name, num_evaluations, population_structure_hashes,generation,tc,saving_convention):
    count = 0
    for i in range(len(structures)):
        structures[i].label = str(generation) +"_"+str(i)
    structures_all = []
    while True:     
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))        
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
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)   
        tc = TerminationCondition(1)
        for structure in structures:
                #structure.compute_fitness()
                structure_bc = (structure.body, structure.connections)
                structure.fitness = run_gappo(structure_bc, tc,(save_path_controller, structure.label),actor_critic,args) 
        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)   #按照structure列表的structure.fitness属性降序排列#
        for structure in structures:
            if structure.is_survivor != True:
                a = copy.deepcopy(structure)
                structures_all.append(a)
        #SAVE RANKING TO FILE
        #temp_path = os.path.join(root_dir, "saved_data", experiment_name, "output.txt")
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()

        ### CHECK EARLY TERMINATION ###
        if count == 5 or num_evaluations == max_evaluations:
            structures_all = sorted(structures_all, key=lambda structure: structure.fitness, reverse=True)
            for i in range(len(structures_all)):
                print("structures_all[i].label:",structures_all[i].label)
            structures_all = structures_all[:pop_size] 
            return structures_all, num_evaluations, population_structure_hashes, generation

        # if num_evaluations == max_evaluations:
        #     print(f'Trained exactly {num_evaluations} robots')
        #     return structures, num_evaluations, population_structure_hashes, generation

        print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)} percent of DESIGNS:\n')
        print(structures[:num_survivors])


        ### CROSSOVER AND MUTATION ###
        # save the survivors
        survivors = structures[:num_survivors]

        fits_txt = [e.fitness for e in structures]
        open(os.path.join(root_dir, "saved_data", experiment_name, "fits.txt"), 'a').write('\t'.join([str(e) for e in [generation] + fits_txt]) + '\n')
        fits = [e.fitness for e in survivors]
        #store survivior information to prevent retraining robots
        for i in range(num_survivors):
            structures[i].is_survivor = True
            structures[i].prev_gen_label = structures[i].label
            # structures[i].label = str(generation+1)+"_"+str(i)

        # for randomly selected survivors, produce children (w mutations)
        num_children = 0
        while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:

            # parent_index = random.sample(range(num_survivors), 1)

            # parent_index = [0]

            parent_index = random.choices(range(num_survivors), fits)
            # IPython.embed()

            child = mutate(survivors[parent_index[0]].body.copy(), mutation_rate = 0.1, num_attempts=50)

            if child != None and hashable(child[0]) not in population_structure_hashes:
                
                # overwrite structures array w new child
                structures[num_survivors + num_children] = Structure(*child, str(generation+1)+"_"+str(num_survivors + num_children))
                structures[num_survivors + num_children].init_pt_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller",
                    "robot_" + str(survivors[parent_index[0]].prev_gen_label) + "_controller" + ".pt")
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1

        structures = structures[:num_children+num_survivors]

        generation += 1
        count += 1
