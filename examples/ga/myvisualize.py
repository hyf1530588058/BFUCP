import os
import torch
import numpy as np
import shutil
import random
import math
import sys     #定义训练的路径#
import pygame
curr_dir = os.path.dirname(os.path.abspath(__file__))  #打印当前文件的绝对路径,获取当前文件上一层目录#
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)   #定义搜索路径的优先级顺序#
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
import time
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, mutate,get_full_connectivity, TerminationCondition, Structure
from ppo.envs import make_vec_envs
import itertools
from ppo import utils
import seaborn as sns
import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
seed = 0 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(1)
# controller_path = os.path.join(root_dir,"saved_data","test","controller","robot_"+str(8)+"_controller"+".pt")   #5.1796   #5.2714
# controller_path = os.path.join(root_dir,"saved_data","test2","controller","robot_"+str(8)+"_controller"+".pt")   #5.42    # 4.2874

# # structure = tuple([np_data["morph"].numpy(),get_full_connectivity(np_data["morph"].numpy())])
# # structure1 = Structure(*structure, 21)
# structure = []
# for key, value in itertools.islice(np_data.items(), 2):
#     structure.append(value)
# structure = list(structure)
# body = np.array([[4., 1. ,4. ,2. ,0.],
#             [2. ,2. ,3., 4., 4.],
#             [1., 2. ,3., 3. ,4.],
#             [4. ,3., 0. ,0. ,1.],
#             [1. ,3., 0.,0., 2.]])
# # print(body)
# connections = get_full_connectivity(body)
# structure[0] = body
# structure[1] = connections
# structure = tuple(structure)
# structure1 = Structure(*structure, 0)
pygame.init()
screen = pygame.display.set_mode((640, 480))  # 设置一个窗口大小
paused = False
name = ["NSLC_Pretrain_upstepper","NSLC_Pretrain_pusher","NSLC_Pretrain_catcher"]

exper_name = "NSLC_Pretrain_carrier"
a = 320

controller_path = os.path.join(root_dir,"saved_data",exper_name,"generation_24","archive_controller","robot_"+str(a)+"_controller"+".pt")
# controller_path = os.path.join(root_dir,"saved_data","RARL_MLP_onlyact_climber","generation_0","controller","robot_"+str(a)+"_controller"+".pt")   #4.0284   #5.4441
actor_critic, obs_rms = torch.load(controller_path, map_location=device)
save_path_structure = os.path.join(root_dir,"saved_data",exper_name,"generation_24","structure",str(a)+".npz")
# save_path_structure = os.path.join(root_dir,"robot_universal","walker","0"+".npz")
structure_data = np.load(save_path_structure)
structure = []

for key, value in structure_data.items():
    structure.append(value)
structure = tuple(structure)
actor_critic.eval()

env = make_vec_envs(
                'Carrier-v0',
                structure,
                5,
                1,
                None,
                None,
                device,
                allow_early_resets=False)

vec_norm = utils.get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

obs = env.reset()
env.render('screen')
time.sleep(0.1)
total_steps = 0
reward_sum = 0
# imgs = []
print(obs.shape)
while total_steps < 2000:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                paused = True
            elif event.key == pygame.K_a:
                paused = False
        elif event.type == pygame.QUIT:
            pygame.quit()
            exit()

    if not paused:        
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs,recurrent_hidden_states, masks, deterministic=True)
    # # print(len(attn))    
    # matrix = attn[0].reshape((-1, attn[1].shape[-1])).cpu()
    # # if total_steps >1000:
    # #     time.sleep(1)
    # # # 使用svd函数计算矩阵的奇异值分解
    # u, s, vt = np.linalg.svd(matrix)
    # # s 奇异值
    # #print("奇异值：")
    # #print(s)
    # squared_sum = np.sum(s**2)
    # #print("每个元素的平方和：", squared_sum)
    # # 寻找最大值
    # max_value = np.max(s)
    # #print("最大值：", max_value)

    # file_path = os.path.join(root_dir,"carrier-attn-M-0")  
    # os.makedirs(file_path, exist_ok=True)
    # f = os.path.join(file_path,"s.txt")
    # with open(f, 'a') as file:
    #     file.write(str(squared_sum) + "\n")       
    # f = os.path.join(file_path,"max.txt")
    # with open(f, 'a') as file:
    #     file.write(str(max_value) + "\n")
    # if total_steps == 356:
    #     time.sleep(60)
        #print("356",attn[0])
    # Obser reward and next obs
    # noise = torch.from_numpy(0.6*np.random.normal(size = action.shape)).to(device)
    # action = action + noise
        obs, reward, done, _ = env.step(action)
        # time.sleep(2)
        masks.fill_(0.0 if (done) else 1.0)
        reward_sum += reward
        total_steps += 1
        # print('reward_sum:',reward_sum)
        # img = env.render(mode='img')
        # imgs.append(img)
        
        env.render('screen')
    # env.get_attr("default_viewer", indices=None)[0].render("screen", hide_grid=True) # 隐藏背景网格
    
    # if done==True:
    #     #print(total_steps)
    #     env.venv.close()
    #     break
print('reward_sum:',reward_sum)
pygame.quit()
env.venv.close()