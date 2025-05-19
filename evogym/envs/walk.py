import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from evogym import *
from evogym.envs import BenchmarkBase

import random
import math
import numpy as np
import os

class WalkingFlatAdv(BenchmarkBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size    #==_adv_bindex: 找到对手施加力的身体部位在列表中的索引
        #num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        #num_robot_points = self.get_relative_pos_obs_matrix_for_robot("robot").size
        num_robot_points = self.get_relative_pos_obs_matrix_for_robot_2("robot").size
        self._adv_bindex = self.get_actuator_indices('robot')   #设置用于施加对手力的身体部位标签，暂时设定为仅针对控制器
        self.adv_action_space = spaces.Box(low= 0.6, high=1.6)

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.pro_action_space = self.action_space
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2 + num_robot_points,), dtype=np.float)

    def _adv_to_xfrc(self, adv_act):    #将对抗动作 adv_act 转换为施加在模型上的力。它首先清零所有施加的力，然后在指定的身体上施加对抗力。
        new_xfrc = self.model.data.xfrc_applied*0.0
        new_xfrc[self._adv_bindex] = np.array([adv_act[0], 0., adv_act[1], 0., 0., 0.])
        self.model.data.xfrc_applied = new_xfrc

    def step(self, action):
        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")
        """
        if hasattr(action, '__dict__'):
            self._adv_to_xfrc(action.adv)     #设置对手的作用力并作用
            a = action.pro
        else:
            a = action
        """
        if hasattr(action, '__dict__'):
            a = action.pro - action.adv
        else:
            a = action
        # step
        done = super().step({'robot': a})   #==self.do_simulation(a, self.frame_skip)

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            # self.get_relative_pos_obs("robot"),
            # self.get_relative_pos_obs_matrix_for_robot("robot"),
            self.get_relative_pos_obs_matrix_for_robot_2("robot"),
            ))

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])
        # y = com_1[1]#求方差，每一步的高度#
        # reward2=0
        # reward3=0
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
            
        # check goal met
        if com_2[0] > 99*self.VOXEL_SIZE:
            done = True
            reward += 1.0
        # reward =reward1+reward2+reward3        
        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            # self.get_relative_pos_obs("robot"),
            # self.get_relative_pos_obs_matrix_for_robot("robot"),
            self.get_relative_pos_obs_matrix_for_robot_2("robot"),
            ))

        return obs
    def update_adversary(self, new_adv_max):
        adv_max_force = new_adv_max
        adv_action_shape = self.adv_action_space.shape[0]
        high_adv = np.ones(adv_action_shape)*adv_max_force
        low_adv = -high_adv
        self.adv_action_space = spaces.Box(low_adv, high_adv)   

class WalkingFlat(BenchmarkBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'Walker-v0.json'))
        self.world.add_from_array('robot', body, 1, 1, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        #num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        #num_robot_points = self.get_relative_pos_obs_matrix_for_robot("robot").size
        num_robot_points = self.get_relative_pos_obs_matrix_for_robot_2("robot").size

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2 + num_robot_points,), dtype=np.float)

    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            # self.get_relative_pos_obs("robot"),
            # self.get_relative_pos_obs_matrix_for_robot("robot"),
            self.get_relative_pos_obs_matrix_for_robot_2("robot"),
            ))

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])
        # y = com_1[1]#求方差，每一步的高度#
        # reward2=0
        # reward3=0
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0
            
        # check goal met
        if com_2[0] > 99*self.VOXEL_SIZE:
            done = True
            reward += 1.0
        # reward =reward1+reward2+reward3        
        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            # self.get_relative_pos_obs("robot"),
            # self.get_relative_pos_obs_matrix_for_robot("robot"),
            self.get_relative_pos_obs_matrix_for_robot_2("robot"),
            ))

        return obs

class SoftBridge(BenchmarkBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'BridgeWalker-v0.json'))
        self.world.add_from_array('robot', body, 2, 5, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        # num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        num_robot_points = self.get_relative_pos_obs_matrix_for_robot_2("robot").size
        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2  + num_robot_points,), dtype=np.float)

    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            # self.get_ort_obs("robot"),
            # self.get_relative_pos_obs("robot"),
            self.get_relative_pos_obs_matrix_for_robot_2("robot"),
            ))

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check goal met
        if com_2[0] > (60)*self.VOXEL_SIZE:
            done = True
            reward += 1.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            # self.get_ort_obs("robot"),
            # self.get_relative_pos_obs("robot"),
            self.get_relative_pos_obs_matrix_for_robot_2("robot"),
            ))

        return obs

class Duck(BenchmarkBase):

    def __init__(self, body, connections=None):

        # make world
        self.world = EvoWorld.from_json(os.path.join(self.DATA_PATH, 'CaveCrawler-v0.json'))
        self.world.add_from_array('robot', body, 1, 2, connections=connections)

        # init sim
        BenchmarkBase.__init__(self, self.world)

        # set action space and observation space
        num_actuators = self.get_actuator_indices('robot').size
        num_robot_points = self.object_pos_at_time(self.get_time(), "robot").size
        self.sight_dist = 5

        self.action_space = spaces.Box(low= 0.6, high=1.6, shape=(num_actuators,), dtype=np.float)
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(2 + num_robot_points + 2*(self.sight_dist*2 +1),), dtype=np.float)

    def step(self, action):

        # collect pre step information
        pos_1 = self.object_pos_at_time(self.get_time(), "robot")

        # step
        done = super().step({'robot': action})

        # collect post step information
        pos_2 = self.object_pos_at_time(self.get_time(), "robot")

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["terrain"], self.sight_dist),
            self.get_ceil_obs("robot", ["terrain"], self.sight_dist),
            ))

        # compute reward
        com_1 = np.mean(pos_1, 1)
        com_2 = np.mean(pos_2, 1)
        reward = (com_2[0] - com_1[0])
        
        # error check unstable simulation
        if done:
            print("SIMULATION UNSTABLE... TERMINATING")
            reward -= 3.0

        # check goal met
        if com_2[0] > (69)*self.VOXEL_SIZE:
            done = True
            reward += 1.0

        # observation, reward, has simulation met termination conditions, debugging info
        return obs, reward, done, {}

    def reset(self):
        
        super().reset()

        # observation
        obs = np.concatenate((
            self.get_vel_com_obs("robot"),
            self.get_relative_pos_obs("robot"),
            self.get_floor_obs("robot", ["terrain"], self.sight_dist),
            self.get_ceil_obs("robot", ["terrain"], self.sight_dist),
            ))

        return obs
