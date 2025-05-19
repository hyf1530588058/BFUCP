import gym
from evogym import sample_robot,get_full_connectivity
import numpy as np
import torch
# import envs from the envs folder and register them
import envs
import os
import itertools
curr_dir = os.path.dirname(os.path.abspath(__file__))  #打印当前文件的绝对路径,获取当前文件上一层目录#
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
if __name__ == '__main__':
    # save_path_structure = os.path.join(root_dir,"examples","robot_universal","robust_advantage_structure",str(9)+".npz")
    save_path_structure = os.path.join(root_dir,"examples","saved_data","NSLC_Pretrain_upstepper","generation_24","robot_universal","walker","0"+".npz")
    np_data = np.load(save_path_structure)
    structure = []
    for key, value in itertools.islice(np_data.items(), 2):
        structure.append(value)
    structure = tuple(structure)
    # create a random robot
    body, connections = sample_robot((5,5))
    body = np.array([[4., 1. ,4. ,2. ,0.],
                [2. ,2. ,3., 4., 4.],
                [1., 2. ,3., 3. ,4.],
                [4. ,3., 0. ,0. ,1.],
                [1. ,3., 0.,0., 2.]])
    # print(body)
    connections = get_full_connectivity(body)
    # make the SimpleWalkingEnv using gym.make and with the robot information
    env = gym.make('Pusher-v0', body=body)
    env.reset()
    print(env.dt)
    # step the environment for 500 iterations
    #for i in range(2000):
    while 1:
        action = env.action_space.sample()
        # length = len(action)
        # a = torch.ones(length)

        ob, reward, done, info = env.step(action)
        env.render(verbose=True)

        # if done:
        #     env.reset()

    env.close()
