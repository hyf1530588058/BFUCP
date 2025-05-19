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
from ppo.myPPOrun1 import run_ppo1
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure
from ppo.myPPOmodel1 import Policy
from ppo.mymodel1 import num_params
from ppo.arguments import get_args
from ppo.envs import make_vec_envs
#device = torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def run_universal1(experiment_name, structure_shape, max_evaluations, train_iters, num_cores):   #此文件与GA文件夹中run文件功能一致#
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
    num_evaluations = 0     #代表的是当前训练需要评估的机器人总数，应当小于给定限制的最大评估数max_evaluations
    label = [41]
    #generate a population
    if not is_continuing:    #在新数据中#
        for i in label:
            #j = i+12
            save_path_structure = os.path.join(root_dir,"robot_universal/walkers_18",str(i) + ".npz")
            np_data = np.load(save_path_structure)    #读取文件
            structure_data = []
            for key, value in np_data.items():  #将读取的原数据添加到新*/的预训练列表中#
                structure_data.append(value)
            structure_data = list(structure_data)
            #structure_data[0] = np.array()            
            structures.append(Structure(*tuple(structure_data), i))  # *号是将列表拆开成两个独立参数：体素数组和连接数组然后传入Structure类当中，label属性是机器人的编号#

    args = get_args()    #在训练最外层初始化模型#
    actor_critic = Policy(
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    print("Num params: {}".format(num_params(actor_critic)))
    #actor_critic.load_state_dict(torch.load(os.path.join(root_dir,"saved_data","49","controller","robot_"+str(4)+"_controller"+".pt"))[0].state_dict())
    print("-=-=-=-=-=-===========start", id(actor_critic))
    while True:
        
        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "structure")   #拼接创建此代幸存者的结构和控制器的保存路径#
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "controller")
        
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
            ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label),actor_critic,args)  #用于传入多进程并行模块的参数，包括机器人结构和标签，终止条件，控制器保存路径,cichu#
            group.add_job(run_ppo1, ppo_args, callback=structure.set_reward)   #对随机生成的机器人添加训练的进程，可以用于获得奖励#
                        
        group.run_jobs(num_cores)  #开始并行训练，每并行训练完num_cores个机器人后就训练下一批num_cores个机器人，每个机器人ppo算法均训练1000轮#

        #not parallel
        #for structure in structures:
        #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)   #按照structure列表的structure.fitness属性降序排列#
        #SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()
        print("-=-=-=-=-=-===========end", id(actor_critic))
        return
    
