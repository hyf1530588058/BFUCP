from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.policies.constant_control_policy import ConstantControlPolicy
import rllab.misc.logger as logger
from rllab.sampler import parallel_sampler
import matplotlib.pyplot as plt
import numpy as np
from .test import test_const_adv, test_rand_adv, test_learnt_adv, test_rand_step_adv, test_step_adv
import pickle
import argparse
import os
import gym
import random
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], 'externals', 'pytorch_a2c_ppo_acktr_gail'))

import numpy as np
import time
from collections import deque
import torch

from ppo import utils
from ppo.arguments import get_args
from ppo.evaluate import evaluate
from ppo.envs import make_vec_envs

from a2c_ppo_acktr import algo          
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.storage import RolloutStorage
import evogym.envs  

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir,'..')
origin_dir =  os.path.join(root_dir,'..')
#from IPython import embed

from ppo.arguments import get_args
args = get_args()

env_name = args.env   #环境的名称
path_length = args.num_steps   #最大回合长度
layer_size = tuple([100,100,100])  #定义神经网络的层结构，表示每层的大小。
ifRender = bool(args.if_render)   #是否渲染环境，0 表示不渲染，1 表示渲染
afterRender = args.after_render   #每经过多少步后进行渲染动画。
n_exps = 1    #运行的训练实例数量。
n_itr = 25    #交替优化的迭代次数。
n_pro_itr = 1   #主角的迭代次数。
n_adv_itr = 1   #对手的迭代次数。
batch_size = 4000   #每次迭代的训练样本数量。
save_every = 100   #每经过多少次迭代保存一次检查点。
n_process = 1   #用于环境采样的并行线程数量。
adv_fraction = 0.25   #应用的最大对抗力的比例。
step_size = 0.01   #    TRPO 算法中的 KL 步长。
gae_lambda = 0.97  #学习者的 GAE（广义优势估计）参数。
save_dir = args.folder  #保存结果的文件夹路径。

## Initializing summaries for the tests ##
const_test_rew_summary = []
rand_test_rew_summary = []
step_test_rew_summary = []
rand_step_test_rew_summary = []
adv_test_rew_summary = []



def run_TRPO(          #此文件功能与ppo原文件夹中的run文件功能一致#
    structure,        #机器人结构，包括体素矩阵和连接矩阵
    termination_condition,    #终止条件
    saving_convention,    #控制器保存路径
    override_env_name = None,
    verbose = True):

    assert (structure == None) == (termination_condition == None) and (structure == None) == (saving_convention == None)    #断言语句，若后续表达式为false则程序中断,此处用于保证所有参数均被输入#

    print(f'Starting training on \n{structure}\nat {saving_convention}...\n')
    if override_env_name:
        args.env_name = override_env_name

    torch.manual_seed(args.seed)   #设置CPU生成随机数种子#
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = args.log_dir
    if saving_convention != None:
        log_dir = os.path.join(saving_convention[0], log_dir, "robot_" + str(saving_convention[1]))
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    #device = torch.device('cpu')

    envs = make_vec_envs(args.env_name, structure, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)
    ## Environment definition ##
    ## The second argument in GymEnv defines the relative magnitude of adversary. For testing we set this to 1.0.
    env = normalize(GymEnv(env_name, adv_fraction))   
    env_orig = normalize(GymEnv(env_name, 1.0))     #    创建两个环境实例：一个用于对抗训练，另一个用于原始环境测试。


    ## Protagonist policy definition ##
    pro_policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=layer_size,
        is_protagonist=True
    )
    pro_baseline = LinearFeatureBaseline(env_spec=env.spec)


    ## Adversary policy definition ##
    adv_policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=layer_size,
        is_protagonist=False
    )
    adv_baseline = LinearFeatureBaseline(env_spec=env.spec)

    ## Initializing the parallel sampler ##
    parallel_sampler.initialize(n_process)   #初始化并行采样器

    ## Optimizer for the Protagonist ##
    pro_algo = TRPO(       #使用 TRPO 算法为主角和对抗者分别设置优化器，配置参数包括环境、策略、基线、批次大小、最大路径长度、迭代次数等。
        env=env,
        pro_policy=pro_policy,
        adv_policy=adv_policy,
        pro_baseline=pro_baseline,
        adv_baseline=adv_baseline,
        batch_size=batch_size,
        max_path_length=path_length,
        n_itr=n_pro_itr,
        discount=0.995,
        gae_lambda=gae_lambda,
        step_size=step_size,
        is_protagonist=True
    )

    ## Optimizer for the Adversary ##
    adv_algo = TRPO(
        env=env,
        pro_policy=pro_policy,
        adv_policy=adv_policy,
        pro_baseline=pro_baseline,
        adv_baseline=adv_baseline,
        batch_size=batch_size,
        max_path_length=path_length,
        n_itr=n_adv_itr,
        discount=0.995,
        gae_lambda=gae_lambda,
        step_size=step_size,
        is_protagonist=False,
        scope='adversary_optim'
    )

    ## Setting up summaries for testing for a specific training instance ##
    pro_rews = []
    adv_rews = []
    all_rews = []
    const_testing_rews = []     #初始化测试结果摘要
    const_testing_rews.append(test_const_adv(env_orig, pro_policy, path_length=path_length))
    rand_testing_rews = []
    rand_testing_rews.append(test_rand_adv(env_orig, pro_policy, path_length=path_length))
    step_testing_rews = []
    step_testing_rews.append(test_step_adv(env_orig, pro_policy, path_length=path_length))
    rand_step_testing_rews = []
    rand_step_testing_rews.append(test_rand_step_adv(env_orig, pro_policy, path_length=path_length))
    adv_testing_rews = []
    adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes   
    ## Beginning alternating optimization ##
    for j in range(num_updates):
        ## Train protagonist
        pro_algo.train()
        # pro_rews += pro_algo.rews; all_rews += pro_algo.rews;
        ## Train Adversary
        adv_algo.train()
        # adv_rews += adv_algo.rews; all_rews += adv_algo.rews;
        ## Test the learnt policies   #进行多种类型的测试，包括常量对抗、随机对抗、步长对抗等，记录每次测试的奖励
        const_testing_rews.append(test_const_adv(env, pro_policy, path_length=path_length))
        rand_testing_rews.append(test_rand_adv(env, pro_policy, path_length=path_length))
        step_testing_rews.append(test_step_adv(env, pro_policy, path_length=path_length))
        rand_step_testing_rews.append(test_rand_step_adv(env, pro_policy, path_length=path_length))
        adv_testing_rews.append(test_learnt_adv(env, pro_policy, adv_policy, path_length=path_length))
       
        if j%100 == 0:    # 定期存储模型
            temp_path_p = os.path.join(args.save_dir, args.algo, args.env_name + ".pt")
            if saving_convention != None:
                temp_path_p = os.path.join(saving_convention[0], "robot_" + str(saving_convention[1]) + "_pro_controller" + ".pt")
            torch.save([     
                pro_policy,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], temp_path_p)   
            temp_path_a = os.path.join(args.save_dir, args.algo, args.env_name + ".pt")
            if saving_convention != None:
                temp_path_a = os.path.join(saving_convention[0], "robot_" + str(saving_convention[1]) + "_adv_controller" + ".pt")
            torch.save([     
                adv_policy,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], temp_path_a)   

        if not termination_condition == None:
            if termination_condition(j):
                ## Shutting down the optimizer ##
                pro_algo.shutdown_worker()
                adv_algo.shutdown_worker()
                ## Updating the test summaries over all training instances
                const_test_rew_summary.append(const_testing_rews)
                rand_test_rew_summary.append(rand_testing_rews)
                step_test_rew_summary.append(step_testing_rews)
                rand_step_test_rew_summary.append(rand_step_testing_rews)
                adv_test_rew_summary.append(adv_testing_rews)             
                if verbose:
                    print(f'{saving_convention} has met termination condition ({j})...terminating...\n')
                return const_test_rew_summary[-1]

