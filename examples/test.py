import os, time
import numpy as np
import shutil
import random
import math
import torch
import sys
import imageio
from ppo.envs import make_vec_envs
from ppo import utils

body = np.array([[4., 3., 0., 2., 2.],
       [3., 4., 4., 4., 1.],
       [3., 0., 1., 3., 4.],
       [3., 4., 3., 0., 3.],
       [4., 3., 0., 0., 3.]])
connection = np.array([[ 0,  0,  1,  3,  3,  4,  5,  5,  6,  7,  7,  8,  8,  9, 10, 12,
        12, 13, 14, 15, 15, 16, 16, 19, 20],
       [ 1,  5,  6,  4,  8,  9,  6, 10,  7,  8, 12,  9, 13, 14, 15, 13,
        17, 14, 19, 16, 20, 17, 21, 24, 21]])
env = make_vec_envs(
                'Walker-v0',
                (body,connection),
                5,
                1,
                None,
                None,
                device='cpu',
                allow_early_resets=False)

while 1:
    # action = env.action_space.sample()
    # ob, reward, done, info = env.step(action)
    env.render('screen')


env.venv.close()