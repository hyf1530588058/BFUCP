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
from ppo.myPPOrun2 import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate, TerminationCondition, Structure
from ppo.myPPOmodel2 import Policy
from ppo.mymodel2 import num_params
from ppo.arguments import get_args
from ppo.envs import make_vec_envs

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def run_universal1(experiment_name, structure_shape, max_evaluations, train_iters, num_cores, iters):   
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
    robot_label = np.random.choice(np.setdiff1d(np.arange(100),[41,54]), max_evaluations, replace=False)
    print("train:",robot_label)
    #generate a population
    if not is_continuing:    #在新数据中#
        for i in robot_label:
            #j = i
            save_path_structure = os.path.join(root_dir,"robot_universal/walkers_18",str(i) + ".npz")
            np_data = np.load(save_path_structure)    #读取文件
            structure_data = []
            for key, value in np_data.items():  #将读取的原数据添加到新的预训练列表中#
                structure_data.append(value)
            structure_data = list(structure_data)
            # structure_data[0] = np.array([[4., 3., 1., 2., 2.],
            #                               [2., 4., 1., 2., 2.],
            #                               [2., 4., 1., 4., 4.],
            #                               [4., 3., 0., 0., 3.],
            #                               [2., 3., 0., 0., 1.]]
            #                             )
            structures.append(Structure(*tuple(structure_data), i))  # *号是将列表拆开成两个独立参数：体素数组和连接数组然后传入Structure类当中，label属性是机器人的编号#

    args = get_args()    #在训练最外层初始化模型#
    actor_critic = Policy(
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)
    print("Num params: {}".format(num_params(actor_critic)))
    #actor_critic.load_state_dict(torch.load(os.path.join(root_dir,"saved_data", "75", "controller","robot_"+str(68)+ "_controller" + ".pt"))[0].state_dict())
    iter_num = 0
    while iter_num < iters:
        
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
            group.add_job(run_ppo, ppo_args, callback=structure.set_reward)   #对随机生成的机器人添加训练的进程，可以用于获得奖励#
                        
        group.run_jobs(num_cores)  #开始并行训练，每并行训练完num_cores个机器人后就训练下一批num_cores个机器人，每个机器人ppo算法均训练1000轮#

        #not parallel
        #for structure in structures:
        #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        if iter_num > 0:
            structures = structures + survivors_structures        

        for structure in structures:
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
            structure.fitness = evaluate(args.num_evals, actor_critic, obs_rms, args.env_name, structure_bc, envs.action_space,args.seed,
                    args.num_processes, eval_log_dir, device)     #将机器人存储的fitness从训练过程中出现的最佳得分更换为当前控制器参数下的直接评估得分#
            

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)   #按照structure列表的structure.fitness属性降序排列#
        #SAVE RANKING TO FILE
        output_dir = os.path.join(root_dir, "saved_data", experiment_name, str(iter_num))
        os.makedirs(output_dir, exist_ok=True)
        temp_path = os.path.join(output_dir, "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\n"
        f.write(out)
        f.close()
        sum_fitness = sum(structure.fitness for structure in structures)
        total_fitness = 0
        worst_k = 0        
        for i,structure in enumerate(structures):
            total_fitness += structure.fitness
            if total_fitness > sum_fitness-total_fitness:
                worst_k = i+1
                break
        survivors_structures = structures[:-worst_k]
        structures = structures[-worst_k:]
        iter_num += 1

    return
    
