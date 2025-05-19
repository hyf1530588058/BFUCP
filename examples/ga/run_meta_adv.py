import os
import torch
import numpy as np
import shutil
import random
import math
from ppo.evaluate import evaluate
import sys     #定义训练的路径#
curr_dir = os.path.dirname(os.path.abspath(__file__))  #打印当前文件的绝对路径,获取当前文件上一层目录#
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)   #定义搜索路径的优先级顺序#
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
import datetime
from ppo.advPPOrun import run_advmetappo_2
# from ppo.advmetamorphmodel import num_params
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure
# from ppo.advmetaPPOmodel import Policy
from ppo.myPPOmodel4 import Policy
from ppo.arguments import get_args
from ppo.envs import make_vec_envs
# from .inverted import inverted_ga
import itertools
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
def run_meta(experiment_name, structure_shape, max_evaluations, train_iters, num_cores):   #此文件与GA文件夹中run文件功能一致#
    print()

    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)

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
    if not is_continuing:    #如果创建初始路径成功或者选择覆盖原数据#
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")  #在元数据中拼接路径#
        
        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))  #创建路径#
        except:
            pass

        f = open(temp_path, "w")    #写入新数据#
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.close()

    else:    #如果选择在原数据上继续进行训练#
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:     #读取原数据中保存的属性数值并赋予#
            if count == 0:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 1:
                max_evaluations = int(line.split()[1])
            if count == 2:
                train_iters = int(line.split()[1])
                tc.change_target(train_iters)
            count += 1

        print(f'Starting training with shape ({structure_shape[0]}, {structure_shape[1]}), ' + 
            f'max evals: {max_evaluations}, train iters {train_iters}.')
        
        f.close()

    structures = []    #记录机器人结构，包括体素矩阵和连接矩阵
    generation = 0
    population_structure_hashes = {}

    if not is_continuing:    #在新数据中#
        for i in range(10):
            save_path_structure = os.path.join(root_dir,"robot_universal/walker",str(i) + ".npz")
            np_data = np.load(save_path_structure)    #读取文件
            structure_data = []
            for key, value in itertools.islice(np_data.items(), 2):  #将读取的原数据添加到新的预训练列表中#
                structure_data.append(value)
            structure_data = tuple(structure_data)           
            structures.append(Structure(*structure_data, i))  # *号是将列表拆开成两个独立参数：体素数组和连接数组然后传入Structure类当中，label属性是机器人的编号#

    args = get_args()    #在训练最外层初始化模型#
    pro_actor_critic = Policy(
        base_kwargs={'recurrent': args.recurrent_policy})
    pro_actor_critic.to(device)
    adv_actor_critic = Policy(
        base_kwargs={'recurrent': args.recurrent_policy})
    adv_actor_critic.to(device)
    # actor_critic.load_state_dict(torch.load(os.path.join(root_dir,"saved_data","robust_advantage_pusher_02","generation_3","controller","robot_"+"2_3_2"+"_controller"+".pt"))[0].state_dict())
    # print("Num params: {}".format(num_params(pro_actor_critic)))
    while True:
        
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

        ### SAVE POPULATION DATA ###
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)   

        ### TRAIN GENERATION

        #better parallel
        group = mp.Group()
        for structure in structures:           
            ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label),pro_actor_critic,adv_actor_critic,args,generation)  #用于传入多进程并行模块的参数，包括机器人结构和标签，终止条件，控制器保存路径,cichu#
            group.add_job(run_advmetappo_2, ppo_args, callback=structure.set_reward)   #对随机生成的机器人添加训练的进程，可以用于获得奖励#
                        
        group.run_jobs(num_cores)  #开始并行训练，每并行训练完num_cores个机器人后就训练下一批num_cores个机器人，每个机器人ppo算法均训练1000轮#

        
        # structures, num_evaluations, population_structure_hashes, generation = inverted_ga(structures, actor_critic, args, max_evaluations, pop_size, experiment_name, num_evaluations, population_structure_hashes, generation)
        # #not parallel
        # #for structure in structures:
        # #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

        # ### COMPUTE FITNESS, SORT, AND SAVE ###
        # for structure in structures:
        #     structure.compute_fitness()

        # structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)   #按照structure列表的structure.fitness属性降序排列#
        # #SAVE RANKING TO FILE
        # temp_path = os.path.join(root_dir, "saved_data", experiment_name, "output.txt")
        # f = open(temp_path, "w")

        # out = ""
        # for structure in structures:
        #     out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        # f.write(out)
        # f.close()

        # if num_evaluations == max_evaluations:
        #     print(f'Trained exactly {num_evaluations} robots')
        return        

    
